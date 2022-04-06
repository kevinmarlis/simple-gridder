import logging
import logging.config
import os
import warnings
from pathlib import Path
import glob

import numpy as np
import xarray as xr
import yaml
from netCDF4 import default_fillvals

import time
from utils import get_date
from datetime import datetime

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import pyresample as pr
    from pyresample.kd_tree import resample_gauss
    from pyresample.utils import check_and_wrap

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), ''))
os.chdir(ROOT_DIR)

logging.config.fileConfig('logs/log.ini', disable_existing_loggers=False)
log = logging.getLogger(__name__)

with open('conf.yaml', 'r') as f:
    config = yaml.safe_load(f)

ALL_DATES = np.arange('2022-01-15', '2022-01-20',
                      config['grid_frequency'], dtype='datetime64[D]')

try:
    INPUT_DIR = Path(config['input_dir'])

    if config['output_dir']:
        GRID_OUTPUT_DIR = Path(config['output_dir'])
    else:
        GRID_OUTPUT_DIR = INPUT_DIR / 'simple_grids'

    DS_NAME = config['ds_name']
    DATE_REGEX = config['filename_date_regex']
except Exception as e:
    print(e)
    log.exception(e)
    exit()


def s6_preprocessing():
    '''
    Function that prepares along track data for gridding by subsetting relevant
    data variables, filtering data through the use of flags or any other constraint,
    and saving processed data in directory.
    '''
    print('Preprocessing Sentinel6 along track data')

    data_dir = INPUT_DIR
    output_dir = INPUT_DIR / 'processed'
    output_dir.mkdir(exist_ok=True)

    for granule in [f for f in os.listdir(data_dir) if f[-2:] == 'nc']:
        data_01_ds = xr.open_dataset(f'{data_dir}{granule}', group='data_01')
        ku_ds = xr.open_dataset(f'{data_dir}{granule}', group='data_01/ku')
        c_ds = xr.open_dataset(f'{data_dir}{granule}', group='data_01/c')

        data_01_das = ['longitude', 'latitude', 'time', 'surface_classification_flag',
                       'rain_flag', 'rad_rain_flag', 'rad_sea_ice_flag']
        ku_das = ['ssha', 'range_ocean_qual']
        c_das = ['range_ocean_qual']

        all_das = []
        for da in data_01_das:
            all_das.append(data_01_ds[da])

        for da in ku_das:
            all_das.append(ku_ds[da])

        for da in c_das:
            temp = c_ds[da]
            temp.name = 'range_ocean_qual_c'
            all_das.append(temp)

        new_ds = xr.Dataset()
        for da in all_das:
            if da.name in ['longitude', 'latitude', 'time']:
                continue
            new_ds[da.name] = da

        flags = ['surface_classification_flag', 'rain_flag',
                 'rad_rain_flag', 'rad_sea_ice_flag', 'range_ocean_qual']

        filtered_ds = xr.Dataset()
        filtered_ds['SSHA'] = new_ds['ssha']
        try:
            for flag in new_ds.data_vars:
                if flag == 'ssha':
                    continue
                filtered_ds['SSHA'] = filtered_ds['SSHA'].where(
                    new_ds[flag] == 0.)
            out_ds = filtered_ds.dropna('time')
            if np.isnan(out_ds['SSHA'].values).all():
                print('all nans', granule)
            datetimeindex = out_ds.indexes['time'].to_datetimeindex()
            out_ds['time'] = datetimeindex

            out_ds['SSHA'] = out_ds['SSHA'].where(abs(out_ds['SSHA']) <= 1.)
            out_ds['SSHA'] = out_ds['SSHA'].where(
                (out_ds['latitude'] > -60) & (abs(out_ds['SSHA']) <= 0.5))
            out_ds = out_ds.dropna('time')
            out_ds.to_netcdf(f'{output_dir}/{granule}')
        except Exception as e:
            log.exception(f'{granule}, {e}')


def ecco_along_track_preprocessing(cycle_granules):
    '''
    Function that prepares along track data for gridding by subsetting relevant
    data variables, filtering data through the use of flags or any other constraint,
    and saving processed data in directory.

    ECCO data only requires variable renaming.
    '''
    print('Preprocessing ECCO along track data')

    processed_granules = []

    for granule in cycle_granules:
        try:
            ds = xr.open_dataset(f'{INPUT_DIR}/{granule}')

            ds = ds.rename_vars(
                {'SSH_at_xy': 'SSHA', 'lat': 'latitude', 'lon': 'longitude'})
            processed_granules.append(ds)
        except Exception as e:
            print(e)
            exit()

    return processed_granules


def tonga_preprocessing(cycle_granules):
    processed_granules = []

    for granule in cycle_granules:
        print(granule)
        try:
            ds = xr.open_dataset(granule, group='data')
            ds = ds.rename_vars(
                {'ssh': 'SSHA', 'lats': 'latitude', 'lons': 'longitude'})
            times = ds.time.values

            lons, lats = check_and_wrap(
                ds.longitude.values.ravel(), ds.latitude.values.ravel())
            ds = xr.Dataset(
                data_vars=dict(
                    SSHA=(['time'], ds.SSHA.values),
                    latitude=(['time'], lats),
                    longitude=(['time'], lons),
                ),
                coords={'time': times}
            )
            ds = ds.where((ds.latitude >= -66) & (ds.latitude <= -40)
                          & (ds.longitude >= -170) & (ds.longitude <= -130), drop=True)
            if 'JASON' in granule and '15' in granule:
                ds = ds.drop_sel(time=[t for t in ds.time.values if t >=
                                       1168850199.86633 and t <= 1168853150.10115])

        except Exception as e:
            print(e)
            log.exception(e)

        processed_granules.append(ds)
    return processed_granules


preprocessers = {"sentinel6": s6_preprocessing,
                 "ecco": ecco_along_track_preprocessing,
                 "tonga": tonga_preprocessing}


def collect_data(start, end):
    '''
    We want to get files in the directory that fall between start and end times
    based on the time in filename. Will need regex to extract time from filename
    '''
    cycle_granules = []
    try:
        all_files = [f for f in glob.glob(
            f'{INPUT_DIR}/**/*{config["file_format"]}', recursive=True)]
        cycle_granules = [f for f in all_files if config['file_format'] in f and
                          get_date(DATE_REGEX, f) <= end and get_date(DATE_REGEX, f) >= start]
        cycle_granules.sort()
    except Exception as e:
        log.exception(e)

    return cycle_granules


def merge_granules(cycle_granules):
    granules = []

    for processed_ds in cycle_granules:
        ds = xr.Dataset(
            data_vars=dict(
                SSHA=(['time'], processed_ds.SSHA.values),
                latitude=(['time'], processed_ds.latitude.values),
                longitude=(['time'], processed_ds.longitude.values),
            )
        )

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
        granules.append(ds)

    cycle_ds = xr.concat((granules), dim='time') if len(
        granules) > 1 else granules[0]
    # cycle_ds = cycle_ds.sortby('time')

    return cycle_ds


def gauss_grid(ssha_nn_obj, global_obj, params):

    tmp_ssha_lons, tmp_ssha_lats = check_and_wrap(ssha_nn_obj['lon'].ravel(),
                                                  ssha_nn_obj['lat'].ravel())

    ssha_grid = pr.geometry.SwathDefinition(
        lons=tmp_ssha_lons, lats=tmp_ssha_lats)

    print('beginning gridding')
    new_vals, _, counts = resample_gauss(ssha_grid, ssha_nn_obj['ssha'],
                                         global_obj['swath'],
                                         radius_of_influence=params['roi'],
                                         sigmas=params['sigma'],
                                         fill_value=np.NaN, neighbours=params['neighbours'],
                                         nprocs=4, with_uncert=True)
    print('completed gridding')
    new_vals_2d = np.zeros_like(global_obj['ds'].landseamask.values) * np.nan

    for i, val in enumerate(new_vals):
        new_vals_2d.ravel()[global_obj['wet'][i]] = val

    counts_2d = np.zeros_like(global_obj['ds'].landseamask.values) * np.nan
    for i, val in enumerate(counts):
        counts_2d.ravel()[global_obj['wet'][i]] = val
    return new_vals_2d, counts_2d


def gridding(cycle_ds, cycle_center):
    # Prepare global map
    global_path = 'ref_files/GPM_IMERG_LandSeaMask.2.nc4'
    global_ds = xr.open_dataset(global_path)

    global_ds = global_ds.transpose('lat', 'lon')

    wet_ins = np.where(global_ds.landseamask.values.ravel() > 50)[0]

    global_lon = global_ds.lon.values
    global_lat = global_ds.lat.values

    # lat = np.arange(-60, -9.9, 0.1)
    # lon = np.arange(-180, -119.9, 0.1)

    global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)
    target_lons_wet = global_lon_m.ravel()[wet_ins]
    target_lats_wet = global_lat_m.ravel()[wet_ins]

    global_swath_def = pr.geometry.SwathDefinition(lons=target_lons_wet,
                                                   lats=target_lats_wet)

    global_obj = {
        'swath': global_swath_def,
        'ds': global_ds,
        'wet': wet_ins
    }

    # Define the 'swath' as the lats/lon pairs of the model grid
    ssha_lon = cycle_ds.longitude.values.ravel()
    ssha_lat = cycle_ds.latitude.values.ravel()
    ssha = cycle_ds.SSHA.values.ravel()

    ssha_lat_nn = ssha_lat[~np.isnan(ssha)]
    ssha_lon_nn = ssha_lon[~np.isnan(ssha)]
    ssha_nn = ssha[~np.isnan(ssha)]

    ssha_nn_obj = {
        'lat': ssha_lat_nn,
        'lon': ssha_lon_nn,
        'ssha': ssha_nn
    }

    if config['gridding_params']:
        params = config['gridding_params']
    else:
        params = {
            'roi': 6e5,  # 6e5
            'sigma': 1e5,
            'neighbours': 500  # 500 for production, 10 for development
        }

    if np.sum(~np.isnan(ssha_nn)) > 0:
        new_vals, counts = gauss_grid(ssha_nn_obj, global_obj, params)
    else:
        raise ValueError('No ssha values.')

    time_seconds = cycle_center.astype('datetime64[s]').astype('int')

    gridded_da = xr.DataArray(new_vals, dims=['latitude', 'longitude'],
                              coords={'latitude': global_lat,
                                      'longitude': global_lon})

    gridded_da = gridded_da.assign_coords(coords={'time': time_seconds})

    gridded_da.name = 'SSHA'
    gridded_ds = gridded_da.to_dataset()

    counts_da = xr.DataArray(counts, dims=['latitude', 'longitude'],
                             coords={'latitude': global_lat,
                                     'longitude': global_lon})
    counts_da = counts_da.assign_coords(coords={'time': time_seconds})

    gridded_ds['counts'] = counts_da

    gridded_ds['mask'] = (['latitude', 'longitude'], np.where(
        global_ds.landseamask.values > 50, 1, 0))

    gridded_ds['mask'].attrs = {'long_name': 'wet/dry boolean mask for grid cell',
                                'comment': '1 for ocean, otherwise 0'}

    gridded_ds['SSHA'].attrs = cycle_ds['SSHA'].attrs
    gridded_ds['SSHA'].attrs['valid_min'] = np.nanmin(
        gridded_ds['SSHA'].values)
    gridded_ds['SSHA'].attrs['valid_max'] = np.nanmax(
        gridded_ds['SSHA'].values)
    gridded_ds['SSHA'].attrs['summary'] = 'Data gridded to 0.5 degree lat lon grid'

    gridded_ds['counts'].attrs = {
        'valid_min': np.nanmin(counts_da.values),
        'valid_max': np.nanmax(counts_da.values),
        'long_name': 'number of data values used in weighting each element in SSHA',
        'source': 'Returned from pyresample resample_gauss function.'
    }

    gridded_ds['latitude'].attrs = cycle_ds['latitude'].attrs
    gridded_ds['longitude'].attrs = cycle_ds['longitude'].attrs

    gridded_ds['time'].attrs = {
        'long_name': 'time',
        'standard_name': 'time',
        'units': 'seconds since 1970-01-01',
        'calendar': 'proleptic_gregorian',
        'comment': 'seconds since 1970-01-01 00:00:00'
    }

    gridded_ds.attrs['gridding_method'] = f'Gridded using pyresample resample_gauss with roi={params["roi"]}, \
            neighbours={params["neighbours"]}'

    # gridded_ds.attrs['source'] = 'Combination of ' + \
    #     ', '.join(sources) + ' along track instruments'

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


def cycle_gridding():
    print(f'Gridding files in {INPUT_DIR}')
    # The main loop
    for date in ALL_DATES:
        start = time.time()

        cycle_start = date - np.timedelta64(4, 'D')
        cycle_end = date + np.timedelta64(5, 'D')
        cycle_center = date

        solr_start = f'{cycle_start}'.replace('-', '')
        solr_end = f'{cycle_end}'.replace('-', '')

        try:
            # Get data within cycle period
            cycle_granules = collect_data(solr_start, solr_end)

            if not cycle_granules:
                print(
                    f'No granules found for cycle centered on {cycle_center}')
                continue

            print(f'Beginning processing of cycle centered on {cycle_center}')
            cycle_granules = preprocessers[DS_NAME](cycle_granules)

            print(f'\tMerging granules for cycle centered on {cycle_center}')
            cycle_ds = merge_granules(cycle_granules)

            print(f'\tGridding cycle centered on {cycle_center}...')
            gridded_ds = gridding(cycle_ds, cycle_center)
            print(f'\tGridding cycle centered on {cycle_center} complete.')

            # Save the gridded cycle
            grid_dir = GRID_OUTPUT_DIR
            grid_dir.mkdir(parents=True, exist_ok=True)
            filename = f'SSHA_gridded_{str(cycle_center)}.nc'
            filepath = grid_dir / filename
            encoding = cycle_ds_encoding(gridded_ds)

            gridded_ds.to_netcdf(filepath, encoding=encoding)
            end = time.time()
            print(f'Total time: {end - start}')
            exit()

        except Exception as e:
            log.exception(e)
            print(e)


if __name__ == '__main__':
    cycle_gridding()
