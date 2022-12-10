from glob import glob
import logging
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import pyresample as pr
    from pyresample.kd_tree import resample_gauss
    from pyresample.utils import check_and_wrap

from conf.global_settings import FILE_FORMAT
import enso_grids


def collect_data(output_dir, start, end):
    def date_filter(f):
        date = f.split('/')[-1].split('.')[0][-8:]
        date = f'{date[:4]}-{date[4:6]}-{date[6:]}'
        date = np.datetime64(date)

        if date >= start and date <= end:
            return True
        return False
    
    def ref_filter(f):
        '''
        Removes MERGED_ALT granules
        '''
        return "MERGED_ALT" not in f

    # Get reference mission granules
    ref_granules = glob(f'{output_dir}/datasets/MERGED_ALT/harvested_granules/**/*{FILE_FORMAT}')
    ref_granules.sort()
    ref_granules = filter(date_filter, ref_granules)

    # Get other granules
    other_granules = glob(f'{output_dir}/datasets/**/**/**/*{FILE_FORMAT}')
    other_granules.sort()

    other_granules = filter(date_filter, other_granules)
    other_granules = filter(ref_filter, other_granules)

    cycle_granules = list(other_granules) + list(ref_granules)
    cycle_granules = sorted(cycle_granules, key=lambda f: f.split('/')[-1].split('.')[0][3:])
    return cycle_granules


def check_updating(output_dir, cycle_granules, date):
    '''
    Compare local files
    '''

    # Check if gridded cycle exists
    grid_path = f'{output_dir}/gridded_cycles/ssha_global_half_deg_{str(date).replace("-", "")}.nc'
    if not os.path.exists(grid_path):
        return True

    grid_mod_time = datetime.fromtimestamp(os.path.getmtime(grid_path))
    # Check if individual granules have been updated
    for granule in cycle_granules:
        mod_time = datetime.fromtimestamp(os.path.getmtime(granule))

        if mod_time > grid_mod_time:
            return True

    return False


def merge_granules(cycle_granules):
    granules = []

    for granule in cycle_granules:
        ds = xr.open_dataset(granule, group='data')
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
        # all_times = ds.time.values
        # seen = set()
        # seen_add = seen.add
        # seen_twice = list(x for x in all_times if x in seen or seen_add(x))
        # if seen_twice:
        #     _, index = np.unique(ds['time'], return_index=True)
        #     ds = ds.isel(time=index)

        granules.append(ds)

    cycle_ds = xr.concat((granules), dim='time', data_vars="all") if len(
        granules) > 1 else granules[0]
    cycle_ds = cycle_ds.sortby('time')
    cycle_ds.to_netcdf('cycle_ds.nc')
    return cycle_ds


def gauss_grid(ssha_nn_obj, global_obj, params):

    tmp_ssha_lons, tmp_ssha_lats = check_and_wrap(ssha_nn_obj['lon'].ravel(),
                                                  ssha_nn_obj['lat'].ravel())

    ssha_grid = pr.geometry.SwathDefinition(
        lons=tmp_ssha_lons, lats=tmp_ssha_lats)
    new_vals, _, counts = resample_gauss(ssha_grid, ssha_nn_obj['ssha'],
                                         global_obj['swath'],
                                         radius_of_influence=params['roi'],
                                         sigmas=params['sigma'],
                                         fill_value=np.NaN, neighbours=params['neighbours'],
                                         nprocs=4, with_uncert=True)

    new_vals_2d = np.full_like(global_obj['ds'].maskC.isel(Z=0).values, np.nan, np.double)
    for i, val in enumerate(new_vals):
        new_vals_2d.ravel()[global_obj['wet'][i]] = val

    counts_2d = np.full_like(global_obj['ds'].maskC.isel(Z=0).values, np.nan, np.double)
    for i, val in enumerate(counts):
        counts_2d.ravel()[global_obj['wet'][i]] = val
    return new_vals_2d, counts_2d


def gridding(cycle_ds, date, sources):

    ref_path = Path().resolve().parent / 'ref_files'

    # Prepare global map
    global_path = ref_path / 'UPDATED_GRID_MASK_latlon.nc'
    global_ds = xr.open_dataset(global_path)

    wet_ins = np.where(global_ds.maskC.isel(Z=0).values.ravel() > 0)[0]

    global_lon = global_ds.longitude.values
    global_lat = global_ds.latitude.values

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

    params = {
        'roi': 6e5,  # 6e5
        'sigma': 1e5,
        'neighbours': 500  # 500 for production, 10 for development
    }

    if np.sum(~np.isnan(ssha_nn)) > 0:
        new_vals, counts = gauss_grid(ssha_nn_obj, global_obj, params)
    else:
        raise ValueError('No ssha values.')

    time_seconds = date.astype('datetime64[s]').astype('int')

    gridded_da = xr.DataArray(new_vals, dims=['latitude', 'longitude'],
                              coords={'longitude': global_lon,
                                      'latitude': global_lat})

    gridded_da = gridded_da.assign_coords(coords={'time': time_seconds})

    gridded_da.name = 'SSHA'
    gridded_ds = gridded_da.to_dataset()

    counts_da = xr.DataArray(counts, dims=['latitude', 'longitude'],
                             coords={'longitude': global_lon,
                                     'latitude': global_lat})
    counts_da = counts_da.assign_coords(coords={'time': time_seconds})

    gridded_ds['counts'] = counts_da

    gridded_ds['mask'] = (['latitude', 'longitude'], np.where(
        global_ds['maskC'].isel(Z=0) == True, 1, 0))

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

    gridded_ds.attrs['gridding_method'] = \
        f'Gridded using pyresample resample_gauss with roi={params["roi"]}, \
            neighbours={params["neighbours"]}'

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


def cycle_gridding(output_dir):
    ALL_DATES = np.arange('1992-10-05', 'now', 7, dtype='datetime64[D]')
    ALL_DATES = np.arange('2012-08-20', 'now', 7, dtype='datetime64[D]')


    failed_grids = []

    for date in ALL_DATES:
        cycle_start = date - np.timedelta64(5, 'D')
        cycle_end = cycle_start + np.timedelta64(9, 'D')

        try:
            cycle_granules = collect_data(output_dir, cycle_start, cycle_end)

            if not cycle_granules or not check_updating(output_dir, cycle_granules, date):
                logging.info(f'No update needed for {date} cycle')
                continue

            logging.info(f'Processing {date} cycle')
            logging.debug(f'\tMerging granules for {date} cycle')
            cycle_ds = merge_granules(cycle_granules)
            sources = list(set([g.split('/datasets/')[1].split('/')[0] for g in cycle_granules]))

            logging.debug(f'\tGridding {date} cycle...')
            gridded_ds = gridding(cycle_ds, date, sources)
            logging.debug(f'\tGridding {date} cycle complete.')

            # Save the gridded cycle
            encoding = cycle_ds_encoding(gridded_ds)

            grid_dir = output_dir / 'gridded_cycles'
            grid_dir.mkdir(parents=True, exist_ok=True)
            filename = f'ssha_global_half_deg_{str(date).replace("-", "")}.nc'
            filepath = grid_dir / filename

            gridded_ds.to_netcdf(filepath, encoding=encoding)

            enso_grids.make_grid(gridded_ds)

        except Exception as e:
            failed_grids.append(date)
            logging.exception(f'\nError while processing cycle {date}. {e}')

    if failed_grids:
        logging.info(f'{len(failed_grids)} grids failed. Check logs')

    return
