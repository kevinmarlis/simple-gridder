import warnings
from datetime import datetime
from pathlib import Path
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module

import numpy as np
import xarray as xr
from conf.global_settings import OUTPUT_DIR
from scipy.interpolate import griddata

warnings.filterwarnings('ignore')

def get_decimal_year(dt: datetime):
    year_start = datetime(dt.year, 1, 1)
    year_end = year_start.replace(year=dt.year+1)
    return dt.year + ((dt - year_start).total_seconds() /  # seconds so far
        float((year_end - year_start).total_seconds()))  # seconds in year

def interp(lats, lons, data):
    array = np.ma.masked_invalid(data)
    xx, yy = np.meshgrid(lons, lats)
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    nn_data = griddata((x1, y1), newarr.ravel(), (xx, yy),  method='nearest')

    grid_x, grid_y = np.meshgrid(np.arange(0.125,360.125, 0.25), np.arange(-89.875,90.125,0.25))

    interpolated_values = griddata((xx.flatten(), yy.flatten()), nn_data.ravel(), (grid_x, grid_y), method='cubic')
    interp_values_2d = np.reshape(interpolated_values, (len(grid_x), len(grid_y[0])))

    return interp_values_2d


def smoothing(ds, date):
    ref_path = Path().resolve().parent / 'ref_files'
    hr_mask_ds = xr.open_dataset(ref_path / 'HR_GRID_MASK_latlon.nc')

    counts = np.where(np.isnan(ds.counts.values), 0, ds.counts.values)
    data = np.where(counts < 400, np.nan, ds.SSHA.values)
    interp_ssha_2d = interp(ds.latitude.values, ds.longitude.values, data)

    interp_ds = xr.Dataset(
        data_vars=dict(
            SSHA=(['latitude', 'longitude'], interp_ssha_2d),
        ),
        coords=dict(
            latitude=(['latitude'], np.arange(-89.875,90.125,0.25)),
            longitude=(['longitude'], np.arange(0.125, 360.125, 0.25)),
            time=(['time'], [date])
        )
    )

    interp_ds.time.encoding['units'] = 'days since 1992-01-01'

    # Do boxcar averaging
    dsr = interp_ds.rolling({'longitude':38, 'latitude':16}, min_periods=1).mean()
    interp_ds.shift
    hr_mask_ds.coords['longitude'] = hr_mask_ds.coords['longitude'] % 360
    hr_mask_ds = hr_mask_ds.sortby(hr_mask_ds.longitude)

    dsr.SSHA.values = np.where(hr_mask_ds.maskC.values == 0, np.nan, dsr.SSHA.values)

    dsr_subset = dsr.sel(latitude=slice(-82,82))

    return dsr_subset

def remove_trends(data, date):
    ref_path = Path().resolve().parent / 'ref_files'

    decimal_year = get_decimal_year(date)
    yr_fraction = decimal_year - date.year
    seas_ds = xr.open_dataset(ref_path / 'trnd_seas_simple_grid.nc')
    seas_ds.coords['Longitude'] = (seas_ds.coords['Longitude']) % 360
    seas_ds = seas_ds.sortby(seas_ds.Longitude)

    cycle_ds = seas_ds.interp({'Month_grid': yr_fraction})
    removed_cycle_data = data - (cycle_ds.Seasonal_SSH.values * 10)
    trend = (decimal_year * seas_ds.SSH_Slope * 10) + (seas_ds.SSH_Offset * 10)
    removed_cycle_trend_data = removed_cycle_data - trend
    return removed_cycle_trend_data

def padding(ds):
    front = ds.sel(longitude=slice(0,10))
    back = ds.sel(longitude=slice(350,360))
    front = front.assign_coords({'longitude': front.longitude.values + 360})
    back = back.assign_coords({'longitude': back.longitude.values - 360})
    padded_ds = xr.merge([back, ds, front])
    return padded_ds

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

def make_grid(ds):
    ds.coords['longitude'] = (ds.coords['longitude']) % 360
    ds = ds.sortby(ds.longitude)
    data = ds.SSHA.values * 1000
    date = datetime.utcfromtimestamp(ds.time.values)
    date_str = datetime.strftime(date, '%Y%m%d')

    removed_cycle_trend_data = remove_trends(data, date)
    ds.SSHA.values = removed_cycle_trend_data

    padded_ds = padding(ds)

    smooth_ds = smoothing(padded_ds, date)
    smooth_ds.SSHA.attrs = ds.SSHA.attrs
    smooth_ds.SSHA.attrs['units'] = 'mm'
    smooth_ds.SSHA.attrs['valid_min'] = np.nanmin(smooth_ds.SSHA.values)
    smooth_ds.SSHA.attrs['valid_max'] = np.nanmax(smooth_ds.SSHA.values)
    smooth_ds.SSHA.attrs['summary'] = 'Data gridded to 0.25 degree grid with boxcar smoothing applied'

    smooth_ds.latitude.attrs = {'long_name': 'latitude', 'standard_name': 'latitude'}
    smooth_ds.longitude.attrs = {'long_name': 'longitude', 'standard_name': 'longitude'}

    encoding = cycle_ds_encoding(smooth_ds)

    fname = f'ssha_enso_{date_str}.nc'
    smooth_ds.to_netcdf(f'{OUTPUT_DIR}/ENSO_grids/{fname}')