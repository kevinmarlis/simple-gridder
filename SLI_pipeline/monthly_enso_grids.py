import logging
from typing import Iterable
import warnings
from datetime import datetime
from pathlib import Path
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module

import numpy as np
import xarray as xr
from conf.global_settings import OUTPUT_DIR
from glob import glob
import os

warnings.filterwarnings('ignore')

def make_monthly_dates(start: str, end: str='now') -> Iterable[str]:
    '''
    Start should be date string of form: %Y-%m-01
    Returns list of monthly dates of form %Y%m
    '''
    dates = [str(d).replace('-', '') for d in np.arange(start, end, 1, dtype='datetime64[M]')]
    return dates

def get_enso_grids_by_date(date: str) -> Iterable[str]:
    files = glob(f'{OUTPUT_DIR}/ENSO_grids/*{date}*.nc')
    files.sort()
    return files

def concat_and_average(files: Iterable[str], date: str) -> xr.Dataset:
    all_data = [xr.open_dataset(f) for f in files]
    ds = xr.concat(all_data, dim='time').mean('time', keep_attrs=True)
    ds.SSHA.attrs['valid_min'] = np.nanmin(ds.SSHA.values)
    ds.SSHA.attrs['valid_max'] = np.nanmax(ds.SSHA.values)
    ds.SSHA.attrs['summary'] = f'Monthly average of {ds.SSHA.attrs["summary"].lower()}'
    ds = ds.assign_coords({'time': [datetime(int(date[:4]), int(date[4:6]), 1)]})
    return ds

def save_ds(ds: xr.Dataset, date: str):
    var_encoding = {'zlib': True,
                    'complevel': 5,
                    'dtype': 'float32',
                    'shuffle': True,
                    '_FillValue': default_fillvals['f8']}
    encoding = {var: var_encoding for var in ds.data_vars}
    encoding['time'] = {'units' : 'days since 1985-01-01'}
    filename = f'monthly_ENSO_{date}.nc'
    logging.info(f'Saving {filename}')
    ds.to_netcdf(f'{OUTPUT_DIR}/monthly_ENSO_grids/{filename}', encoding = encoding)

if __name__ == '__main__':
    dates = make_monthly_dates('2023-05-01')
    logging.info(f'Making monthly ENSO grids for {dates}.')
    
    for date in dates:
        print(date, type(date))
        files = get_enso_grids_by_date(date)

        if len(files) < 4:
            logging.info(f'{date} contains insufficient ENSO grids. Skipping.')
            continue
        
        ds = concat_and_average(files, date)
        save_ds(ds, date)
        