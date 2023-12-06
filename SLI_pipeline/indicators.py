from glob import glob
import logging
import os
from typing import Iterable
import warnings
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
import xarray as xr
from netCDF4 import default_fillvals

from conf.global_settings import OUTPUT_DIR

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    from pyresample.utils import check_and_wrap

REF_DIR = Path().resolve().parent / 'ref_files'
PATTERNS = ['enso', 'pdo', 'iod']

class Pattern():
    name: str
    pattern: xr.Dataset
    ann_cyc: xr.Dataset
    
    def __init__(self, pattern: str) -> None:
        self.name = pattern
        ann_ds = xr.open_dataset(REF_DIR / 'ann_pattern.nc')
        self.pattern = xr.open_dataset(REF_DIR / f'{pattern}_pattern_and_index.nc')
        self.pattern = self.pattern.rename({'Latitude': 'latitude',
                                            'Longitude': 'longitude'})
        self.ann_cyc = self.get_ann_cyc(ann_ds)
    
    def get_ann_cyc(self, ann_ds):
        geo_bounds = [float(self.pattern.latitude[0].values),
                    float(self.pattern.latitude[-1].values),
                    float(self.pattern.longitude[0].values),
                    float(self.pattern.longitude[-1].values)]
        return ann_ds.sel(Latitude=slice(geo_bounds[0], geo_bounds[1]),
                               Longitude=slice(geo_bounds[2], geo_bounds[3]))

class Global():
    global_ds: xr.Dataset
    global_dsm: xr.Dataset
    gmsl_da: xr.DataArray
    
    def __init__(self, date: str, ds: xr.Dataset) -> None:
        self.global_ds = xr.open_dataset(REF_DIR / 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc')
        ct = np.datetime64(date)

        # Area mask the cycle data
        global_dam = ds.where(self.global_ds.maskC.isel(Z=0) > 0)['SSHA']
        global_dam = global_dam.where(global_dam)
        global_dam.name = 'SSHA_GLOBAL'
        global_dam.attrs['comment'] = 'Global SSHA land masked'
        self.global_dsm = global_dam.to_dataset()

        # Spatial Mean
        self.gmsl_da = calc_spatial_mean(global_dam, self.global_ds, ct)

        global_dam_removed_mean = global_dam - self.gmsl_da.values
        global_dam_removed_mean.attrs['comment'] = 'Global SSHA with global spatial mean removed'
        self.global_dsm['SSHA_GLOBAL_removed_global_spatial_mean'] = global_dam_removed_mean

        # Linear Trend
        self.trend = calc_linear_trend(ds)
        self.global_dsm['SSHA_GLOBAL_linear_trend'] = self.trend

        global_dam_detrended = global_dam - self.trend
        global_dam_detrended.attrs['comment'] = 'Global SSHA with linear trend removed'
        self.global_dsm['SSHA_GLOBAL_removed_linear_trend'] = global_dam_detrended

        if 'Z' in self.global_dsm.data_vars:
            self.global_dsm = self.global_dsm.drop_vars('Z')
            pass

def validate_counts(ds: xr.Dataset, threshold: float = 0.9) -> bool:
    '''
    Checks if counts average is above threshold value.
    '''
    counts = ds.sel(latitude=slice(-66, 66))['counts'].values
    mean = np.nanmean(counts)

    if mean > threshold * 500:
        return True
    return False


def calc_linear_trend(cycle_ds: xr.Dataset) -> float:
    trend_ds = xr.open_dataset(REF_DIR / 'BH_offset_and_trend_v0_new_grid.nc')

    cycle_time = cycle_ds.time.values.astype('datetime64[D]')
    time_diff = (cycle_time - np.datetime64('1992-10-02')).astype(np.int32) * 24 * 60 * 60

    trend = time_diff * \
        trend_ds['BH_sea_level_trend_meters_per_second'] + \
        trend_ds['BH_sea_level_offset_meters']

    return trend


def calc_spatial_mean(global_dam, ecco_latlon_grid, ct):
    global_dam_slice = global_dam.sel(latitude=slice(-66, 66))
    ecco_latlon_grid_slice = ecco_latlon_grid.sel(latitude=slice(-66, 66))

    nzp = np.where(~np.isnan(global_dam_slice), 1, np.nan)
    area_nzp = np.sum(nzp * ecco_latlon_grid_slice.area)

    spatial_mean = float(np.nansum(global_dam_slice * ecco_latlon_grid_slice.area) / area_nzp)
    spatial_mean_da = xr.DataArray(spatial_mean, coords={'time': ct}, attrs=global_dam.attrs)

    spatial_mean_da.name = 'spatial_mean'
    spatial_mean_da.attrs['comment'] = 'Global SSHA spatial mean'
    return spatial_mean_da


def calc_climate_index(agg_ds: xr.Dataset, pattern: Pattern):
    '''
    '''

    center_time = agg_ds.time.values

    # determine its month
    agg_ds_center_mon = int(str(center_time)[5:7])

    pattern_field = pattern.pattern[f'{pattern.name}_pattern'].values

    ssha_da = agg_ds[f'SSHA_{pattern.name}_removed_global_linear_trend']

    # remove the monthly mean pattern from the gridded ssha
    # now ssha_anom is w.r.t. seasonal cycle and MDT
    ssha_anom = ssha_da.values - pattern.ann_cyc.ann_pattern.sel(month=agg_ds_center_mon).values/1e3

    # set ssha_anom to nan wherever the original pattern is nan
    ssha_anom = np.where(~np.isnan(pattern.pattern[f'{pattern.name}_pattern']), ssha_anom, np.nan)

    # extract out all non-nan values of ssha_anom, these are going to
    # be the points that we fit
    nonnans = ~np.isnan(ssha_anom)
    ssha_anom_to_fit = ssha_anom[nonnans]

    # do the same for the pattern
    pattern_to_fit = pattern_field[nonnans]/1e3

    # just for fun extract out same points from ssha, we'll see if
    # removing the monthly climatology makes much of a difference
    ssha_to_fit = ssha_da.copy(deep=True)
    ssha_to_fit = ssha_da.values[nonnans]

    X = np.vstack(np.array(pattern_to_fit))

    # Good old Gauss
    B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), ssha_anom_to_fit.T)
    offset = 0
    index = B_hat[0]

    # now minimize ssha_to_fit
    B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), ssha_to_fit.T)
    offset_b = 0
    index_b = B_hat[0]

    LS_result = [offset, index, offset_b, index_b]

    lats = ssha_da.latitude.values
    lons = ssha_da.longitude.values

    ssha_anom = xr.DataArray(ssha_anom, dims=['latitude', 'longitude'],
                             coords={'longitude': lons, 'latitude': lats})

    return LS_result, center_time, ssha_anom


def save_files(date, output_dir, indicator_ds):
    ds_and_paths = []

    fp_date = date.replace('-', '_')
    cycle_indicators_path = output_dir / 'cycle_indicators'
    cycle_indicators_path.mkdir(parents=True, exist_ok=True)
    indicator_output_path = cycle_indicators_path / f'{fp_date}_indicator.nc'
    ds_and_paths.append((indicator_ds, indicator_output_path))

    var_encoding = {'zlib': True,
                     'complevel': 5,
                     'dtype': 'float32',
                     'shuffle': True,
                     '_FillValue': default_fillvals['f8']}
    coord_encoding = {'_FillValue': None, 'dtype': 'float32', 'complevel': 6}

    for ds, path in ds_and_paths:
        coord_encodings = {coord: coord_encoding for coord in ds.coords}
        var_encodings = {var: var_encoding for var in ds.data_vars}
        encoding = {**coord_encodings, **var_encodings}

        ds.to_netcdf(path, encoding=encoding)
        ds.close()

    return


def recalculate_indicators(grids: Iterable[str]) -> bool:
    '''
    
    '''
    # Check if we need to recalculate indicators
    data_path = f'{OUTPUT_DIR}/indicator/indicators.nc'
    if not os.path.exists(data_path):
        return True
    
    backup_dir = Path(f'{OUTPUT_DIR}/indicator/backups')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    ind_mod_time = datetime.fromtimestamp(os.path.getmtime(data_path))
    for grid in grids:
        if datetime.fromtimestamp(os.path.getmtime(grid)) >= ind_mod_time:
            # Copy old indicator file as backup
            try:
                logging.info('Making backup of existing indicator file.\n')
                backup_path = f'{backup_dir}/indicator_{ind_mod_time}.nc'
                copyfile(data_path, backup_path)
            except Exception as e:
                logging.exception(f'Error creating indicator backup: {e}')
            return True
    return False


def cycle_processing(cycle: str, output_dir):   
    cycle_ds = xr.open_dataset(cycle)
    cycle_ds.close()

    date = cycle.split('_')[-1][:8]
    date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'

    # Skip this grid if it's missing too much data
    if not validate_counts(cycle_ds):
        logging.exception(f'Too much data missing from {date} cycle. Skipping.')
        return

    logging.info(f'Calculating index values for {date}')

    global_data = Global(date, cycle_ds)
    
    all_indicators = []

    # Do the actual index calculation per pattern
    for pattern_name in PATTERNS:
        pattern = Pattern(pattern_name)
        pattern_lons, pattern_lats = check_and_wrap(pattern.pattern.longitude.values,
                                                    pattern.pattern.latitude.values)

        agg_da = global_data.global_dsm['SSHA_GLOBAL_removed_linear_trend'].sel(longitude=pattern_lons, latitude=pattern_lats)
        agg_da.name = f'SSHA_{pattern.name}_removed_global_linear_trend'

        agg_ds = agg_da.to_dataset()
        agg_ds.attrs = cycle_ds.attrs

        index_calc, ct, ssha_anom = calc_climate_index(agg_ds, pattern)

        # Handle indicators and offsets
        indicator_da = xr.DataArray(index_calc[1], coords={'time': ct})
        indicator_da.name = f'{pattern.name}_index'
        all_indicators.append(indicator_da)

        offsets_da = xr.DataArray(index_calc[0], coords={'time': ct})
        offsets_da.name = f'{pattern.name}_offset'
        all_indicators.append(offsets_da)
    
    all_indicators.append(global_data.gmsl_da)
    indicator_ds = xr.merge(all_indicators)
    indicator_ds = indicator_ds.expand_dims(time=[indicator_ds.time.values])

    save_files(date, output_dir, indicator_ds)


def indicators():
    """
    This function calculates indicator values for each regridded cycle. Those are
    saved locally to avoid overloading memory. All locally saved indicator files 
    are combined into a single netcdf spanning the entire 1992 - NOW time period.
    """
    # Get all gridded cycles
    grids = glob(f'{OUTPUT_DIR}/gridded_cycles/*.nc')
    grids.sort()
    
    # ONLY PROCEED IF THERE ARE CYCLES NEEDING CALCULATING
    if not recalculate_indicators(grids):
        logging.info('No regridded cycles modified since last index calculation.')
        return True

    logging.info('Calculating new index values for cycles.')

    daily_indicator_dir = OUTPUT_DIR / 'indicator' / 'daily'
    daily_indicator_dir.mkdir(parents=True, exist_ok=True)

    # Calculate indicators for each updated (re)gridded cycle
    for cycle in grids:
        date = cycle.split('_')[-1][:8]
        date = f'{date[:4]}_{date[4:6]}_{date[6:8]}'
        indicator_path = daily_indicator_dir / 'cycle_indicators' / f'{date}_indicator.nc'

        if os.path.exists(indicator_path) and os.path.getmtime(cycle) < os.path.getmtime(indicator_path):
            logging.info(f'Indicator value for {date} already computed.')
            continue
        try:
            cycle_processing(cycle, daily_indicator_dir)
        except Exception as e:
            logging.exception(f'Error processing cycle {cycle}. {e}')

    logging.info('Cycle index calculation complete. ')

    # Combine DAILY indicator files
    try:
        indicator_dir = OUTPUT_DIR / 'indicator'

        daily_path = indicator_dir / f'DAILY/cycle_indicators'
        daily_files = [x for x in daily_path.glob('*.nc') if x.is_file()]
        daily_files.sort()
        
        logging.info('Opening all daily indicator files')
        all_ds = [xr.open_dataset(f) for f in daily_files]

        logging.info(f'Concatenating daily indicator files')
        indicators_ds = xr.concat(all_ds, dim='time')

        logging.info('Saving indicator file')
        indicators_ds.to_netcdf(indicator_dir / 'indicators.nc')

    except Exception as e:
        logging.exception(e)
        return False

    return True