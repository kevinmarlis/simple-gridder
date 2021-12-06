from collections import defaultdict
from datetime import datetime
import logging
import logging.config
from pathlib import Path

import h5py  # need to import for xarray to open hdf5 properly
import numpy as np
import pyresample as pr
import xarray as xr
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module
from pyresample.kd_tree import resample_gauss
from pyresample.utils import check_and_wrap

from utils import file_utils, solr_utils

logs_path = 'SLI_pipeline/logs/'
logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)


def collect_data(start, end):
    # Get reference mission granules (prefer MERGED_ALT when available)
    fq = ['type_s:granule', 'dataset_s:(MERGED_ALT OR JASON_3)', 'harvest_success_b:true',
          f'date_dt:[{start} TO {end}]']
    ref_granules = solr_utils.solr_query(
        fq, sort='date_dt asc, dataset_s desc')

    ref_data = defaultdict(list)

    for g in ref_granules:
        if not ref_data[g['date_dt']]:
            ref_data[g['date_dt']] = g

    # Get other mission granules
    fq = ['type_s:granule', '-dataset_s:MERGED_ALT', '-dataset_s:JASON_3', 'harvest_success_b:true',
          f'date_dt:[{start} TO {end}]']
    other_data = solr_utils.solr_query(fq, sort='date_dt asc')

    cycle_granules = list(ref_data.values()) + other_data
    cycle_granules = sorted(cycle_granules, key=lambda d: d['date_dt'])

    return cycle_granules


def check_updating(cycle_granules, date):
    # Check if gridded cycle exists
    fq = ['type_s:gridded_cycle',
          'processing_success_b:true', f'date_dt:"{date}"']
    r = solr_utils.solr_query(fq)

    if not r or not r[0]['processing_success_b']:
        return True

    gridded_time = r[0]['processing_time_dt']

    # Check if individual granules have been updated
    for granule in cycle_granules:
        granule_time = granule['download_time_dt']

        if granule_time > gridded_time:
            return True

    return False


def merge_granules(cycle_granules):
    granules = []

    for granule in cycle_granules:
        ds = xr.open_dataset(granule['granule_file_path_s'], group='data')
        ds = xr.Dataset(
            data_vars=dict(
                SSHA=(['time'], ds.ssh.values),
                latitude=(['time'], ds.lats.values),
                longitude=(['time'], ds.lons.values),
                time=(['time'], ds.time.values)
            )
        )

        ds.time.attrs = {
            'long_name': 'time',
            'standard_name': 'time',
            'units': 'seconds since 1985-01-01',
            'calendar': 'gregorian',
        }

        ds.latitude.attrs = {
            'long_name': 'latitude',
            'standard_name': 'latitude',
            'units': 'degrees_north',
            'comment': 'Positive latitude is North latitude, negative latitude is South latitude. FillValue pads the reference orbits to have same length'
        }

        ds.longitude.attrs = {
            'long_name': 'longitude',
            'standard_name': 'longitude',
            'units': 'degrees_east',
            'comment': 'East longitude relative to Greenwich meridian. FillValue pads the reference orbits to have same length'
        }

        ds.SSHA.attrs = {
            'long_name': 'sea surface height anomaly',
            'standard_name': 'sea_surface_height_above_sea_level',
            'units': 'm',
            'valid_min': np.nanmin(ds.SSHA.values),
            'valid_max': np.nanmax(ds.SSHA.values),
            'comment': 'Sea level determined from satellite altitude - range - all altimetric corrections',
        }

        # Check for duplicate time values. Drop if they are true duplicates
        all_times = ds.time.values
        seen = set()
        seen_add = seen.add
        seen_twice = list(x for x in all_times if x in seen or seen_add(x))
        if seen_twice:
            _, index = np.unique(ds['time'], return_index=True)
            ds = ds.isel(time=index)

        granules.append(ds)

    cycle_ds = xr.concat((granules), dim='time') if len(
        granules) > 1 else granules[0]
    cycle_ds = cycle_ds.sortby('time')

    return cycle_ds


def gridding(cycle_ds, date, sources):

    ref_files_path = Path().resolve() / 'SLI_pipeline' / 'ref_files'

    # Prepare global map
    global_path = ref_files_path / 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc'
    global_ds = xr.open_dataset(global_path)

    wet_ins = np.where(global_ds.maskC.isel(Z=0).values.ravel() > 0)[0]

    global_lon = global_ds.longitude.values
    global_lat = global_ds.latitude.values

    global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)
    target_lons_wet = global_lon_m.ravel()[wet_ins]
    target_lats_wet = global_lat_m.ravel()[wet_ins]

    global_swath_def = pr.geometry.SwathDefinition(lons=target_lons_wet,
                                                   lats=target_lats_wet)

    # Define the 'swath' as the lats/lon pairs of the model grid
    ssha_lon = cycle_ds.longitude.values.ravel()
    ssha_lat = cycle_ds.latitude.values.ravel()
    ssha = cycle_ds.SSHA.values.ravel()

    ssha_lat_nn = ssha_lat[~np.isnan(ssha)]
    ssha_lon_nn = ssha_lon[~np.isnan(ssha)]
    ssha_nn = ssha[~np.isnan(ssha)]

    if np.sum(~np.isnan(ssha_nn)) > 0:
        tmp_ssha_lons, tmp_ssha_lats = check_and_wrap(ssha_lon_nn.ravel(),
                                                      ssha_lat_nn.ravel())

        ssha_grid = pr.geometry.SwathDefinition(
            lons=tmp_ssha_lons, lats=tmp_ssha_lats)

        roi = 6e5  # 6e5
        sigma = 1e5
        neighbours = 10  # 500 for production, 10 for development

        new_vals = resample_gauss(ssha_grid, ssha_nn,
                                  global_swath_def,
                                  radius_of_influence=roi,
                                  sigmas=sigma,
                                  fill_value=np.NaN, neighbours=neighbours)

        new_vals_2d = np.zeros_like(global_ds.area.values) * np.nan
        for i, val in enumerate(new_vals):
            new_vals_2d.ravel()[wet_ins[i]] = val
        new_vals = new_vals_2d

    gridded_da = xr.DataArray(new_vals, dims=['latitude', 'longitude'],
                              coords={'longitude': global_lon,
                                      'latitude': global_lat})

    gridded_da = gridded_da.assign_coords(coords={'time': date})

    gridded_da.name = 'SSHA'
    gridded_ds = gridded_da.to_dataset()

    gridded_ds['mask'] = (['latitude', 'longitude'], np.where(
        global_ds['maskC'].isel(Z=0) == True, 1, 0))

    gridded_ds['mask'].attrs = {'long_name': 'wet/dry boolean mask for grid cell',
                                'comment': '1 for ocean, otherwise 0'}

    # gridded_ds['SSHA'].attrs = cycle_ds['SSHA'].attrs
    gridded_ds['SSHA'].attrs['valid_min'] = np.nanmin(
        gridded_ds['SSHA'].values)
    gridded_ds['SSHA'].attrs['valid_max'] = np.nanmax(
        gridded_ds['SSHA'].values)
    gridded_ds['SSHA'].attrs['summary'] = 'Data gridded to 0.5 degree lat lon grid'

    gridded_ds['latitude'].attrs = cycle_ds['latitude'].attrs
    gridded_ds['longitude'].attrs = cycle_ds['longitude'].attrs

    gridded_ds.attrs['gridding_method'] = \
        f'Gridded using pyresample resample_gauss with roi={roi}, neighbours={neighbours}'

    gridded_ds.attrs['source'] = 'Combination of ' + \
        ', '.join(sources) + ' along track instruments'

    return gridded_ds


def cycle_ds_encoding(cycle_ds):
    """
    Generates encoding dictionary used for saving the cycle netCDF file.
    The measures gridded dataset (1812) has additional units encoding requirements.

    Params:
        cycle_ds (Dataset): the Dataset object
        ds_name (str): the name of the dataset (used to check if dataset is 1812)
        center_date (datetime): used to set the units encoding in the 1812 dataset

    Returns:
        encoding (dict): the encoding dictionary for the cycle_ds Dataset object
    """

    var_encoding = {'zlib': True,
                    'complevel': 5,
                    'dtype': 'float32',
                    'shuffle': True,
                    '_FillValue': default_fillvals['f8']}
    var_encodings = {var: var_encoding for var in cycle_ds.data_vars}

    coord_encoding = {}
    for coord in cycle_ds.coords:
        if 'Time' in coord:
            coord_encoding[coord] = {'_FillValue': None,
                                     'zlib': True,
                                     'contiguous': False,
                                     'shuffle': False}

        if 'Lat' in coord or 'Lon' in coord:
            coord_encoding[coord] = {'_FillValue': None, 'dtype': 'float32'}

    encoding = {**coord_encoding, **var_encodings}
    return encoding


def run_status():
    """
    Determines processing status by number of failed and successful cycle documents on Solr.
    Updates dataset document on Solr with status message

    Return:
        processing_status (str): overall processing status
    """

    processing_status = 'All cycles successfully gridded'

    # Query for failed regridded cycle documents
    fq = ['type_s:gridded_cycle', 'processing_success_b:false']
    failed_processing = solr_utils.solr_query(fq)

    if failed_processing:
        processing_status = 'No cycles successfully gridded (all failed or no granules to process)'

        # Query for successful cycle documents
        fq = ['type_s:gridded_cycle', 'processing_success_b:true']
        successful_processing = solr_utils.solr_query(fq)

        if successful_processing:
            processing_status = f'{len(failed_processing)} cycles failed'

    return processing_status


def cycle_gridding(output_dir):
    ALL_DATES = np.arange('1992-10-05', 'now', 7, dtype='datetime64[D]')
    output_dir = Path('/Users/marlis/Developer/SLI/dev_output')
    date_regex = '%Y-%m-%dT%H:%M:%S'

    # The main loop
    for date in ALL_DATES:
        cycle_start = date - np.timedelta64(5, 'D')
        cycle_end = cycle_start + np.timedelta64(9, 'D')

        solr_date = f'{date}T00:00:00Z'
        solr_start = f'{cycle_start}T00:00:00Z'
        solr_end = f'{cycle_end}T00:00:00Z'

        try:
            cycle_granules = collect_data(solr_start, solr_end)

            if not cycle_granules or not check_updating(cycle_granules, solr_date):
                print(f'No update needed for {date} cycle')
                continue

            print(f'Beginning processing of {date} cycle')
            print(f'\tMerging granules for {date} cycle')
            cycle_ds = merge_granules(cycle_granules)
            sources = list(set([g['dataset_s'] for g in cycle_granules]))

            print(f'\tGridding {date} cycle...')
            gridded_ds = gridding(cycle_ds, date, sources)
            print(f'\tGridding {date} cycle complete.')

            # Save the gridded cycle
            grid_dir = output_dir / 'gridded_cycles'
            grid_dir.mkdir(parents=True, exist_ok=True)
            filename = f'ssha_global_half_deg_{str(date).replace("-", "")}.nc'
            filepath = grid_dir / filename
            encoding = cycle_ds_encoding(gridded_ds)

            gridded_ds.to_netcdf(filepath, encoding=encoding)

            checksum = file_utils.md5(filepath)
            file_size = filepath.stat().st_size
            processing_success = True
        except Exception as e:
            # log.exception(f'\nError while processing cycle {date}. {e}')
            print(e)
            filename = ''
            filepath = ''
            checksum = ''
            file_size = 0
            processing_success = False

        item = {}
        item['type_s'] = 'gridded_cycle'
        item['date_dt'] = solr_date
        item['start_dt'] = solr_start
        item['end_dt'] = solr_end
        item['cycle_length_i'] = 10
        item['filename_s'] = filename
        item['filepath_s'] = str(filepath)
        item['checksum_s'] = checksum
        item['file_size_l'] = file_size
        item['processing_success_b'] = processing_success
        item['processing_time_dt'] = datetime.utcnow().strftime(date_regex)

        # Check if gridded cycle already exists in Solr
        fq = ['type_s:gridded_cycle', f'date_dt:"{solr_date}"']
        r = solr_utils.solr_query(fq)
        if r:
            item['id'] == r[0]['id']

        resp = solr_utils.solr_update([item], True)
        if resp.status_code == 200:
            print(
                '\tSuccessfully created or updated Solr cycle documents')
        else:
            print('\tFailed to create Solr cycle documents')

    return run_status()
