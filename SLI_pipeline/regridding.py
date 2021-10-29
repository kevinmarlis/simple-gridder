"""
"""

import logging
import pickle
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyresample as pr
import xarray as xr
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module
from pyresample.kd_tree import resample_gauss
from pyresample.utils import check_and_wrap

from utils import file_utils, solr_utils, grid_utils


warnings.filterwarnings("ignore")

logs_path = 'SLI_pipeline/logs/'
logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)


# 'sort': 'date_dt asc'}


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
    fq = ['type_s:regridded_cycle', 'processing_success_b:false']
    failed_processing = solr_utils.solr_query(fq)

    if failed_processing:
        processing_status = 'No cycles successfully processed (all failed or no granules to process)'

        # Query for successful cycle documents
        fq = ['type_s:regridded_cycle', 'processing_success_b:true']
        successful_processing = solr_utils.solr_query(fq)

        if successful_processing:
            processing_status = f'{len(failed_processing)} cycles failed'

    return processing_status


def regridder(cycle_ds, data_type, output_dir, method='gaussian', ats=[], neighbours=500):

    date_regex = '%Y-%m-%dT%H:%M:%S'

    mapping_dir = output_dir / 'mappings'
    mapping_dir.mkdir(parents=True, exist_ok=True)

    ref_files_path = Path().resolve() / 'SLI_pipeline' / 'ref_files'

    global_path = ref_files_path / 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc'
    global_ds = xr.open_dataset(global_path)

    wet_ins = np.where(global_ds.maskC.isel(Z=0).values.ravel() > 0)[0]

    global_mapping_dir = mapping_dir / 'global'
    global_mapping_dir.mkdir(parents=True, exist_ok=True)

    global_lon = global_ds.longitude.values
    global_lat = global_ds.latitude.values

    global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)
    target_lons_wet = global_lon_m.ravel()[wet_ins]
    target_lats_wet = global_lat_m.ravel()[wet_ins]

    global_swath_def = pr.geometry.SwathDefinition(lons=target_lons_wet,
                                                   lats=target_lats_wet)

    if data_type == 'gridded':

        global_attrs = cycle_ds.attrs
        var_attrs = cycle_ds['SSHA'].attrs

        source_indices_fp = global_mapping_dir / 'half_deg_global_source_indices.p'
        num_sources_fp = global_mapping_dir / 'half_deg_global_num_sources.p'

        if not source_indices_fp.exists() or not num_sources_fp.exists():
            # Create index mappings from 1812 cycle to ECCO grid
            cycle_lons = cycle_ds.longitude.values
            cycle_lats = cycle_ds.latitude.values

            temp_pattern_lons, temp_pattern_lats = check_and_wrap(cycle_lons,
                                                                  cycle_lats)

            cycle_lons_m, cycle_lats_m = np.meshgrid(temp_pattern_lons,
                                                     temp_pattern_lats)

            cycle_swath_def = pr.geometry.SwathDefinition(lons=cycle_lons_m,
                                                          lats=cycle_lats_m)

            global_grid_radius = np.sqrt(
                global_ds.area.values.ravel())/2*np.sqrt(2)

            global_grid_radius_wet = global_grid_radius.ravel()[wet_ins]

            # Used for development
            # neighbours = 1

            source_indices_within_target_radius_i, num_source_indices_within_target_radius_i, nearest_source_index_to_target_i = grid_utils.find_mappings_from_source_to_target(cycle_swath_def,
                                                                                                                                                                                global_swath_def,
                                                                                                                                                                                global_grid_radius_wet,
                                                                                                                                                                                100,
                                                                                                                                                                                20e3,
                                                                                                                                                                                neighbours=neighbours)
            with open(source_indices_fp, "wb") as f:
                pickle.dump(source_indices_within_target_radius_i, f)
            with open(num_sources_fp, "wb") as f:
                pickle.dump(num_source_indices_within_target_radius_i, f)

        else:
            # Load up the existing index mappings
            with open(source_indices_fp, "rb") as f:
                source_indices_within_target_radius_i = pickle.load(f)

            with open(num_sources_fp, "rb") as f:
                num_source_indices_within_target_radius_i = pickle.load(f)

        cycle_vals = cycle_ds.sel(
            time=cycle_ds.time.values[0]).SSHA.values.T
        cycle_vals_1d = cycle_vals.ravel()

        new_vals = np.zeros_like(global_ds.area.values) * np.NaN

        print('\tMapping source to target grid.')
        for i in range(len(num_source_indices_within_target_radius_i)):
            if num_source_indices_within_target_radius_i[i] != 0:

                new_vals.ravel()[wet_ins[i]] = sum(cycle_vals_1d[source_indices_within_target_radius_i[i]]
                                                   ) / num_source_indices_within_target_radius_i[i]

        valid_keys = [k for k in cycle_ds['latitude'].attrs.keys()
                      if 'valid' in k]
        for k in valid_keys:
            del cycle_ds['latitude'].attrs[k]
            del cycle_ds['longitude'].attrs[k]

    # Along track data
    else:

        instr_in_cycle = [ds.attrs['original_dataset_short_name']
                          for ds in ats]
        global_attrs = cycle_ds.attrs
        for key in [attr for attr in global_attrs.keys() if 'original' in attr]:
            global_attrs.pop(key)

        global_attrs['source'] = 'Combination of ' + \
            ', '.join(instr_in_cycle) + ' along track instruments'

        if method == 'gaussian':
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
                neighbours = 500  # 500

                new_vals = resample_gauss(ssha_grid, ssha_nn,
                                          global_swath_def,
                                          radius_of_influence=roi,
                                          sigmas=sigma,
                                          fill_value=np.NaN, neighbours=neighbours)

                new_vals_2d = np.zeros_like(global_ds.area.values) * np.nan
                for i, val in enumerate(new_vals):
                    new_vals_2d.ravel()[wet_ins[i]] = val
                new_vals = new_vals_2d

        else:
            cycle_lons = cycle_ds.longitude.values
            cycle_lats = cycle_ds.latitude.values

            temp_pattern_lons, temp_pattern_lats = check_and_wrap(cycle_lons,
                                                                  cycle_lats)

            cycle_lons_m, cycle_lats_m = np.meshgrid(temp_pattern_lons,
                                                     temp_pattern_lats)

            cycle_swath_def = pr.geometry.SwathDefinition(lons=cycle_lons_m,
                                                          lats=cycle_lats_m)

            global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)
            target_lons_wet = global_lon_m.ravel()[wet_ins]
            target_lats_wet = global_lat_m.ravel()[wet_ins]

            global_swath_def = pr.geometry.SwathDefinition(lons=target_lons_wet,
                                                           lats=target_lats_wet)

            global_grid_radius = np.sqrt(
                global_ds.area.values.ravel())/2*np.sqrt(2)

            global_grid_radius_wet = global_grid_radius.ravel()[wet_ins]

            # Used for development
            # neighbours = 1

            source_indices_within_target_radius_i, num_source_indices_within_target_radius_i, nearest_source_index_to_target_i = grid_utils.find_mappings_from_source_to_target(cycle_swath_def,
                                                                                                                                                                                global_swath_def,
                                                                                                                                                                                global_grid_radius_wet,
                                                                                                                                                                                100,
                                                                                                                                                                                20e3,
                                                                                                                                                                                neighbours=neighbours)

            cycle_vals = cycle_ds.SSHA.values.T
            cycle_vals_1d = cycle_vals.ravel()

            new_vals = np.zeros_like(global_ds.area.values) * np.nan

            print('\tMapping source to target grid.')
            for i in range(len(num_source_indices_within_target_radius_i)):
                if num_source_indices_within_target_radius_i[i] != 0:

                    new_vals.ravel()[wet_ins[i]] = sum(cycle_vals_1d[source_indices_within_target_radius_i[i]]
                                                       ) / num_source_indices_within_target_radius_i[i]

    regridded_da = xr.DataArray(new_vals, dims=['latitude', 'longitude'],
                                coords={'longitude': global_lon,
                                        'latitude': global_lat})

    regridded_da = regridded_da.assign_coords(
        coords={'time': np.datetime64(cycle_ds.cycle_center)})

    regridded_da.name = 'SSHA'
    regridded_ds = regridded_da.to_dataset()

    regridded_ds['mask'] = (['latitude', 'longitude'], np.where(
        global_ds['maskC'].isel(Z=0) == True, 1, 0))

    regridded_ds['mask'].attrs = {'long_name': 'wet/dry boolean mask for grid cell',
                                  'comment': '1 for ocean, otherwise 0'}

    regridded_ds.attrs = cycle_ds.attrs

    regridded_ds['SSHA'].attrs = cycle_ds['SSHA'].attrs
    regridded_ds['SSHA'].attrs['valid_min'] = np.nanmin(
        regridded_ds['SSHA'].values)
    regridded_ds['SSHA'].attrs['valid_max'] = np.nanmax(
        regridded_ds['SSHA'].values)
    regridded_ds['SSHA'].attrs['summary'] = 'Data gridded to 0.5 degree lat lon grid'

    regridded_ds['latitude'].attrs = cycle_ds['latitude'].attrs
    regridded_ds['longitude'].attrs = cycle_ds['longitude'].attrs

    if data_type == 'along_track' and method == 'gaussian':
        regridded_ds.attrs['gridding_method'] = \
            f'Gridded using pyresample resample_gauss with roi={roi}, neighbours={neighbours}'
    else:
        regridded_ds.attrs['gridding_method'] = \
            f'Gridded using find_mappings_from_source_to_target with neighbours={neighbours}'

    regridded_ds.attrs['comment'] = 'Gridded using ECCO V4r4 0.5 degree lat lon grid'

    return regridded_ds


def regridding(output_dir, reprocess):
    """
    Indicator pipeline uses three data sources: MEASuRES 1812, Jason 3, and Sentinel 3B.
    This function combines the two along track datasets on a single 0.5 degree latlon
    grid. It also regrids the gridded MEASuRES data to the same 0.5 degree latlon grid. 
    """

    date_regex = '%Y-%m-%dT%H:%M:%S'
    version = 1.0

    # ======================================================
    # Regrid measures cycles
    # ======================================================
    print('\nRegridding MEaSUREs cycles\n')

    existing_regridded_measures_cycles = defaultdict(list)
    fq = ['type_s:regridded_cycle', 'original_data_type_s:gridded']
    for cycle in solr_utils.solr_query(fq, 'date_dt asc'):
        start_date_key = cycle['start_date_dt'][:10]
        existing_regridded_measures_cycles[start_date_key].append(cycle)

    fq = ['type_s:cycle', 'processing_success_b:true', 'data_type_s:gridded']
    solr_gridded_cycles = solr_utils.solr_query(fq, 'date_dt asc')

    gridded_cycles = defaultdict(list)
    for cycle in solr_gridded_cycles:
        date_key = cycle['start_date_dt'][:10]
        gridded_cycles[date_key].append(cycle)

    for date, cycle in gridded_cycles.items():
        cycle = cycle[0]
        # only regrid if cycle was modified
        update = date not in existing_regridded_measures_cycles.keys() or \
            existing_regridded_measures_cycles[date][0]['processing_success_b'] == False or \
            cycle['processing_time_dt'] > existing_regridded_measures_cycles[date][0]['processing_time_dt']

        if update:
            try:
                print(f'Regridding MEaSUREs cycle {date}')
                cycle_ds = xr.open_dataset(cycle['filepath_s'])
                regridded_ds = regridder(cycle_ds, 'gridded', output_dir)

                regrid_dir = output_dir / 'regridded_cycles/MEASURES_1812'
                regrid_dir.mkdir(parents=True, exist_ok=True)

                center_date = cycle["center_date_dt"]

                filename = f'ssha_global_half_deg_{center_date[:10].replace("-", "_")}.nc'
                global_fp = regrid_dir / filename
                encoding = cycle_ds_encoding(regridded_ds)

                regridded_ds.to_netcdf(global_fp, encoding=encoding)

                # Determine checksum and file size
                checksum = file_utils.md5(global_fp)
                file_size = global_fp.stat().st_size
                processing_success = True
            except:
                log.exception(f'\nError while processing cycle {date}')
                filename = ''
                global_fp = ''
                checksum = ''
                file_size = 0
                processing_success = False

            item = cycle
            item.pop('id')
            item.pop('dataset_s')
            item.pop('_version_')
            item['type_s'] = 'regridded_cycle'
            item['combination_s'] = 'MEASURES_1812'
            item['filename_s'] = filename
            item['filepath_s'] = str(global_fp)
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['processing_success_b'] = processing_success
            item['processing_time_dt'] = datetime.utcnow().strftime(date_regex)
            item['processing_version_f'] = version
            item['checksum_s'] = checksum
            item['original_data_type_s'] = item.pop('data_type_s')

            if date in existing_regridded_measures_cycles.keys():
                item['id'] = existing_regridded_measures_cycles[date][0]['id']

            resp = solr_utils.solr_update([item], True)
            if resp.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')
            else:
                print('\tFailed to create Solr cycle documents')

        else:
            # print(f'\tNo updates to regridded MEaSUREs cycle {date}')
            pass

    # ======================================================
    # Regrid along track cycles
    # ======================================================
    print('\nRegridding along track cycles\n')

    existing_regridded_at_cycles = {}
    fq = ['type_s:regridded_cycle', 'combination_s:DAILY']
    for cycle in solr_utils.solr_query(fq, 'date_dt asc'):
        start_date_key = cycle['start_date_dt'][:10]
        existing_regridded_at_cycles[start_date_key] = cycle

    # Start with J3 cycles
    fq = ['type_s:cycle', 'processing_success_b:true', 'dataset_s:JASON_3']
    j3_cycles = solr_utils.solr_query(fq, 'date_dt asc')

    for j3_cycle in j3_cycles:
        j3_ds = xr.open_dataset(j3_cycle['filepath_s'])

        at_cycles = [j3_ds]

        j3_start = j3_cycle['start_date_dt']
        j3_end = j3_cycle['end_date_dt']

        # Find S3 cycle with time overlap
        # J3 start_date >= S3 start_date and J3 start_date <= S3 end_date
        # S3 start_date >= J3 start_date and J3 end_date <= S3 end_date
        fq = ['type_s:cycle', 'processing_success_b:true', 'dataset_s:SENTINEL_3B',
              f'start_date_dt:["1992-01-01T00:00:00Z" TO {j3_start}]', f'end_date_dt:{{{j3_start} TO NOW]']
        s3_cycles = solr_utils.solr_query(fq, 'date_dt asc')

        update = False

        if j3_start[:10] in existing_regridded_at_cycles.keys():
            existing_regrid_meta = existing_regridded_at_cycles[j3_start[:10]]

            # Determine if J3 or S3 cycles have been updated
            if j3_cycle['processing_time_dt'] > existing_regrid_meta['processing_time_dt']:
                update = True

            if s3_cycles:
                for s3 in s3_cycles:
                    if s3['processing_time_dt'] > existing_regrid_meta['processing_time_dt']:
                        update = True

            if not existing_regrid_meta['processing_success_b']:
                update = True
        else:
            update = True

        if not update:
            print(f'\tNo updates to regridded DAILY cycle {j3_start}')
        else:

            print(f'Regridding DAILY cycle {j3_start}')
            try:
                s3_data = []
                if s3_cycles:
                    for s3 in s3_cycles:
                        s3_ds = xr.open_dataset(s3['filepath_s'])

                        if j3_start[-1] == 'Z':
                            temp_start = j3_start[:-1]
                            temp_end = j3_end[:-1]
                        else:
                            temp_start = j3_start
                            temp_end = j3_end

                        s3_slice_ds = s3_ds.sel(
                            time=slice(temp_start, temp_end))

                        s3_data.append(s3_slice_ds)

                    #   Only need one S3 metadata added to list
                    at_cycles.append(s3_ds)

                all_data = [j3_ds]

                if s3_data:
                    s3_merged = xr.merge(s3_data)
                    all_data.append(s3_merged)

                all_data_ds = xr.concat(all_data, 'time')
                all_data_ds = all_data_ds.sortby('time')

                regridded_ds = regridder(
                    all_data_ds, 'along_track', output_dir, ats=at_cycles)

                regrid_dir = output_dir / 'regridded_cycles/DAILY_test'
                regrid_dir.mkdir(parents=True, exist_ok=True)

                filename = f'ssha_global_half_deg_{j3_start[:10].replace("-", "")}.nc'
                global_fp = regrid_dir / filename
                encoding = cycle_ds_encoding(regridded_ds)

                regridded_ds.to_netcdf(global_fp, encoding=encoding)

                # Determine checksum and file size
                checksum = file_utils.md5(global_fp)
                file_size = global_fp.stat().st_size
                processing_success = True
            except Exception as e:
                log.exception(
                    f'\nError while processing cycle {j3_start}. {e}')
                filename = ''
                global_fp = ''
                checksum = ''
                file_size = 0
                processing_success = False

            item = {}
            item['type_s'] = 'regridded_cycle'
            item['combination_s'] = 'DAILY'
            item['start_date_dt'] = j3_cycle['start_date_dt']
            item['center_date_dt'] = j3_cycle['center_date_dt']
            item['end_date_dt'] = j3_cycle['end_date_dt']
            item['cycle_length_i'] = j3_cycle['cycle_length_i']
            item['filename_s'] = filename
            item['filepath_s'] = str(global_fp)
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['processing_success_b'] = processing_success
            item['processing_time_dt'] = datetime.utcnow().strftime(
                date_regex)
            item['processing_version_f'] = version
            item['original_data_type_s'] = j3_cycle['data_type_s']

            if j3_cycle['start_date_dt'][:10] in existing_regridded_at_cycles.keys():
                item['id'] = existing_regridded_at_cycles[j3_cycle['start_date_dt'][:10]]['id']
            resp = solr_utils.solr_update([item], True)
            if resp.status_code == 200:
                print(
                    '\tSuccessfully created or updated Solr cycle documents')
            else:
                print('\tFailed to create Solr cycle documents')
    return run_status()
