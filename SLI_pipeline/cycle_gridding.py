import functools
import logging
import os
from typing import Iterable
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd
from netCDF4 import default_fillvals # pylint: disable=no-name-in-module
import yaml

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import pyresample as pr
    from pyresample.kd_tree import resample_gauss
    from pyresample.utils import check_and_wrap

from conf.global_settings import DATA_DIR, OUTPUT_DIR
import enso_grids

REF_PATH = Path().resolve().parent / 'ref_files'
ROI = 6e5
SIGMA = 1e5
NEIHBOURS = 500

class Target():
    global_ds: xr.Dataset
    wet_ins: np.ndarray
    target_def: pr.geometry.SwathDefinition
    
    def __init__(self) -> None:
        self.global_ds = xr.open_dataset(REF_PATH / 'UPDATED_GRID_MASK_latlon.nc')
        self.wet_ins = np.where(self.global_ds.maskC.isel(Z=0).values.ravel() > 0)[0]

        global_lon_m, global_lat_m = np.meshgrid(self.global_ds.longitude.values, 
                                                 self.global_ds.latitude.values)
        self.target_def = pr.geometry.SwathDefinition(global_lon_m.ravel()[self.wet_ins], 
                                                      global_lat_m.ravel()[self.wet_ins])


class Source():
    ssha: np.ndarray
    ssha_nn: np.ndarray
    source_def: pr.geometry.SwathDefinition
    
    def __init__(self, ds: xr.Dataset) -> None:
        self.ssha = ds.SSHA.values
        self.ssha_nn = self.ssha[~np.isnan(self.ssha)]
        
        ssha_lat_nn = ds.latitude.values.ravel()[~np.isnan(self.ssha)]
        ssha_lon_nn = ds.longitude.values.ravel()[~np.isnan(self.ssha)]

        self.source_def = pr.geometry.SwathDefinition(*check_and_wrap(ssha_lon_nn.ravel(), 
                                                                     ssha_lat_nn.ravel()))


class Datasets():
    configs: dict
    
    def __init__(self):
        with open(Path(f'conf/datasets.yaml'), "r") as stream:
            config = yaml.load(stream, yaml.Loader)
        self.configs = {c['ds_name']: c for c in config}
   
ds_configs = Datasets()

def get_valid_sats(start: np.datetime64, end: np.datetime64) -> Iterable[str]:
    '''
    Returns list of satellites whose coverage falls within 10 day window
    '''
    valid_sats = []
    for k,v in ds_configs.configs.items():
        config_start = np.datetime64(v.get('start'))
        config_end = np.datetime64('today') if v.get('end') == 'now' else np.datetime64(v.get('end'))
        latest_start = max(start, config_start)
        earliest_end = min(end, config_end)
        delta = (earliest_end - latest_start) + 1
        if max(0, delta):
            valid_sats.append(k)
    return valid_sats

def collect_data(start: np.datetime64, end: np.datetime64) -> Iterable[str]:   
    '''
    Collects valid filepaths for a given 10 day window
    '''
    window_dates = [str(d).replace("-","") for d in np.arange(start, end, 1, dtype='datetime64[D]')]
    window_granules = []
    for sat in get_valid_sats(start, end):
        window_granules.extend([f'{DATA_DIR}/{sat}/{filename}' for filename in os.listdir(f'{DATA_DIR}/{sat}') if filename[-11:-3] in window_dates])
    window_granules = sorted(window_granules, key=lambda f: f.split('/')[-1].split('_')[-1][3:])
    return window_granules


def check_updating(cycle_granules: Iterable[str], date: np.timedelta64) -> bool:
    '''
    Checks if window requires (re)gridding by comparing source granules with
    gridded file.
    '''
    
    # Check if gridded cycle exists
    grid_path = f'{OUTPUT_DIR}/gridded_cycles/ssha_global_half_deg_{str(date).replace("-", "")}.nc'
    if not os.path.exists(grid_path):
        return True

    grid_mod_time = datetime.fromtimestamp(os.path.getmtime(grid_path))
    # Check if individual granules have been updated
    for granule in cycle_granules:
        mod_time = datetime.fromtimestamp(os.path.getmtime(granule))
        if mod_time > grid_mod_time:
            return True
    return False


def apply_s6_correction(ds: xr.Dataset, filename: str) -> xr.Dataset:
    '''
    Applies S6 radiometer corrections to data where relevant
    '''
    correction_file = REF_PATH / 'S6_radiometer_additive_correction.csv'
    df = pd.read_csv(correction_file)

    filename = filename.split('ssh')[-1].split('.')[0]
    file_date = ''.join(filter(str.isdigit, filename))

    try:
        correction_value = float(df[df.index==file_date].value.values)
    except:
        correction_value = 0

    ds.SSHA.values = ds.SSHA.values + correction_value

    return ds


def drop_bad_passes(ds: xr.Dataset) -> xr.Dataset:
    '''
    Removes known bad passes from source data prior to gridding
    '''
    with open(Path(f'conf/datasets.yaml'), "r") as stream:
        config = yaml.load(stream, yaml.Loader)
    configs = {c['ds_name']: c for c in config}

    bad_passes = configs[ds.attrs['source']].get('bad_passes', [])
    ds = functools.reduce(lambda x,y: x.where(x.track_id != y, drop=True), bad_passes, ds)
    ds = ds.drop_vars('track_id')
    return ds


def merge_granules(cycle_granules: Iterable[str]) -> xr.Dataset:
    '''
    Opens and merges all collected source granules within a given 10 day window
    '''
    granules = []

    for granule in cycle_granules:
        ds = xr.open_dataset(granule, group='data')
        ds = xr.Dataset(
            data_vars=dict(
                SSHA=(['time'], ds.ssh.values),
                latitude=(['time'], ds.lats.values),
                longitude=(['time'], ds.lons.values),
                time=(['time'], ds.time.values),
                track_id=(['time'], ds.track_id.values)
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
        ds.attrs['source'] = granule.split('/')[-2]
        
        # Apply temporary sentinel 6A corrections
        if 'SNTNL-6A' in granule:
            ds = apply_s6_correction(ds, granule)
            
        # Drop known bad passes
        ds = drop_bad_passes(ds)

        granules.append(ds)

    cycle_ds = xr.concat((granules), dim='time', data_vars="all") if len(granules) > 1 else granules[0]
    cycle_ds = cycle_ds.sortby('time')
    return cycle_ds

def make_ds(resampled_data: np.ndarray, counts: np.ndarray, target: Target, date: np.datetime64, sources: Iterable[str]) -> xr.Dataset:
    '''
    Converts regridded data (and counts array) to xarray Dataset object with attributes
    '''
    data_2d = np.full_like(target.global_ds.maskC.isel(Z=0).values, np.nan, np.double)
    for i, val in enumerate(resampled_data):
        data_2d.ravel()[target.wet_ins[i]] = val
        
    da = xr.DataArray(data_2d, dims=['latitude', 'longitude'],
                              coords={'longitude': target.global_ds.longitude.values,
                                      'latitude': target.global_ds.latitude.values})
    time_seconds = date.astype('datetime64[s]').astype('int')
    da = da.assign_coords(coords={'time': time_seconds})
    da.name = 'SSHA'
    ds = da.to_dataset()

    counts_2d = np.full_like(target.global_ds.maskC.isel(Z=0).values, np.nan, np.double)
    for i, val in enumerate(counts):
        counts_2d.ravel()[target.wet_ins[i]] = val
    
    ds['counts'] = (['latitude', 'longitude'], counts_2d)

    ds['counts'].attrs = {
        'valid_min': np.nanmin(ds.counts.values),
        'valid_max': np.nanmax(ds.counts.values),
        'long_name': 'number of data values used in weighting each element in SSHA',
        'source': 'Returned from pyresample resample_gauss function.'
    }

    ds['mask'] = (['latitude', 'longitude'], np.where(target.global_ds.maskC.isel(Z=0) == True, 1, 0))
    ds['mask'].attrs = {
        'long_name': 'wet/dry boolean mask for grid cell',
        'comment': '1 for ocean, otherwise 0'
        }

    ds['SSHA'].attrs = {
        'long_name': 'sea surface height anomaly',
        'standard_name': 'sea_surface_height_above_sea_level',
        'units': 'm',
        'valid_min': np.nanmin(ds.SSHA.values),
        'valid_max': np.nanmax(ds.SSHA.values),
        'comment': 'Sea level determined from satellite altitude - range - all altimetric corrections',
        'summary': 'Data gridded to 0.5 degree lat lon grid'
        }

    ds['latitude'].attrs = {
        'long_name': 'latitude',
        'standard_name': 'latitude',
        'units': 'degrees_north',
        'comment': 'Positive latitude is North latitude, negative latitude is South latitude. FillValue pads the reference orbits to have same length'
        }
    ds['longitude'].attrs = {
        'long_name': 'longitude',
        'standard_name': 'longitude',
        'units': 'degrees_east',
        'comment': 'East longitude relative to Greenwich meridian. FillValue pads the reference orbits to have same length'
        }

    ds['time'].attrs = {
        'long_name': 'time',
        'standard_name': 'time',
        'units': 'seconds since 1970-01-01',
        'calendar': 'proleptic_gregorian',
        'comment': 'seconds since 1970-01-01 00:00:00'
    }

    ds.attrs['gridding_method'] = f'Gridded using pyresample resample_gauss with roi={ROI}, neighbours={NEIHBOURS}'
    ds.attrs['source'] = f'Combination of {", ".join(sources)} along track instruments'
    return ds


def gridding(cycle_ds: xr.Dataset, date: np.datetime64, sources: Iterable[str]) -> xr.Dataset:
    '''
    Performs gridding using pyresample's resample_gauss function.
    '''
    source = Source(cycle_ds)
    target = Target()
    
    if np.isnan(source.ssha_nn).all():
        raise ValueError(f'No valid SSHA values for {str(date)}')

    resampled_data, stddev, counts = resample_gauss(source.source_def, source.ssha_nn, target.target_def, 
                                                    ROI, SIGMA, NEIHBOURS, fill_value=np.NaN, nprocs=4, 
                                                    with_uncert=True)
    
    ds = make_ds(resampled_data, counts, target, date, sources) 
    return ds


def cycle_ds_encoding(ds: xr.Dataset) -> dict:
    """
    Generates encoding dictionary used for saving the cycle netCDF file.
    """
    var_encoding = {'zlib': True,
                    'complevel': 5,
                    'dtype': 'float32',
                    'shuffle': True,
                    '_FillValue': default_fillvals['f8']}
    var_encodings = {var: var_encoding for var in ds.data_vars}

    coord_encoding = {}
    for coord in ds.coords:
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
    '''
    Creates gridded netCDFs for each 10 day window occuring every 7 days
    '''
    ALL_DATES = np.arange('1992-10-05', 'now', 7, dtype='datetime64[D]')

    failed_grids = []
    for date in ALL_DATES[:-1]:
        cycle_start = date - np.timedelta64(5, 'D')
        cycle_end = cycle_start + np.timedelta64(9, 'D')

        try:
            cycle_granules = collect_data(cycle_start, cycle_end)
            if not cycle_granules:
                logging.info(f'No granules found for {date} cycle')
                continue

            if not check_updating(cycle_granules, date):
                logging.info(f'No update needed for {date} cycle')
                continue

            logging.info(f'Processing {date} cycle')
            cycle_ds = merge_granules(cycle_granules)
            sources = list(set([g.split('/')[-2].split('/')[0] for g in cycle_granules]))

            gridded_ds = gridding(cycle_ds, date, sources)

            # Save the gridded cycle
            encoding = cycle_ds_encoding(gridded_ds)

            grid_dir = OUTPUT_DIR / 'gridded_cycles'
            grid_dir.mkdir(parents=True, exist_ok=True)
            filename = f'ssha_global_half_deg_{str(date).replace("-", "")}.nc'
            filepath = grid_dir / filename

            gridded_ds.to_netcdf(filepath, encoding=encoding)

            # enso_grids.make_grid(gridded_ds)

        except Exception as e:
            failed_grids.append(date)
            logging.exception(f'Error while processing cycle {date}. {e}')

    if failed_grids:
        logging.info(f'{len(failed_grids)} grids failed. Check logs')
