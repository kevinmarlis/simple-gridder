"""

"""
import logging
import warnings
from datetime import datetime
from pathlib import Path
from shutil import copyfile

import numpy as np
import xarray as xr
from netCDF4 import default_fillvals

with warnings.catch_warnings():
    warnings.simplefilter('ignore', UserWarning)
    import pyresample as pr
    from pyresample.utils import check_and_wrap

from utils import file_utils, solr_utils

logging.config.fileConfig(f'logs/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)


def calc_linear_trend(ref_dir, cycle_ds):
    trend_ds = xr.open_dataset(
        ref_dir / 'BH_offset_and_trend_v0_new_grid.nc')

    cycle_time = cycle_ds.time.values.astype('datetime64[D]')

    time_diff = (cycle_time - np.datetime64('1992-10-02')
                 ).astype(np.int32) * 24 * 60 * 60

    trend = time_diff * \
        trend_ds['BH_sea_level_trend_meters_per_second'] + \
        trend_ds['BH_sea_level_offset_meters']

    return trend


def calc_spatial_mean(global_dam, ecco_latlon_grid, ct):
    global_dam_slice = global_dam.sel(latitude=slice(-66, 66))
    ecco_latlon_grid_slice = ecco_latlon_grid.sel(
        latitude=slice(-66, 66))

    nzp = np.where(~np.isnan(global_dam_slice), 1, np.nan)
    area_nzp = np.sum(nzp * ecco_latlon_grid_slice.area)

    spatial_mean = float(np.nansum(global_dam_slice *
                         ecco_latlon_grid_slice.area) / area_nzp)

    spatial_mean_da = xr.DataArray(spatial_mean, coords={'time': ct},
                                   attrs=global_dam.attrs)

    spatial_mean_da.name = 'spatial_mean'
    spatial_mean_da.attrs['comment'] = 'Global SSHA spatial mean'
    return spatial_mean_da


def calc_climate_index(agg_ds, pattern, pattern_ds, ann_cyc_in_pattern):
    """

    Params:
        agg_ds (Dataset): the aggregated cycle Dataset object
        pattern (str): the name of the pattern
        pattern_ds (Dataset): the actual pattern object
        ann_cyc_in_pattern (Dict):
        weights_dir (Path): the Path to the directory containing the stored pattern weights
    Returns:
        LS_result (List[float]):
        center_time (Datetime):
    """

    center_time = agg_ds.time.values

    # determine its month
    agg_ds_center_mon = int(str(center_time)[5:7])

    pattern_field = pattern_ds[pattern][f'{pattern}_pattern'].values

    ssha_da = agg_ds[f'SSHA_{pattern}_removed_global_linear_trend']

    # remove the monthly mean pattern from the gridded ssha
    # now ssha_anom is w.r.t. seasonal cycle and MDT
    ssha_anom = ssha_da.values - \
        ann_cyc_in_pattern[pattern].ann_pattern.sel(
            month=agg_ds_center_mon).values/1e3

    # set ssha_anom to nan wherever the original pattern is nan
    ssha_anom = np.where(
        ~np.isnan(pattern_ds[pattern][f'{pattern}_pattern']), ssha_anom, np.nan)

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
    B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T),
                      ssha_anom_to_fit.T)
    offset = 0
    index = B_hat[0]

    # now minimize ssha_to_fit
    B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T),
                      ssha_to_fit.T)
    offset_b = 0
    index_b = B_hat[0]

    LS_result = [offset, index, offset_b, index_b]

    lats = ssha_da.latitude.values
    lons = ssha_da.longitude.values

    ssha_anom = xr.DataArray(ssha_anom, dims=['latitude', 'longitude'],
                             coords={'longitude': lons,
                                     'latitude': lats})

    return LS_result, center_time, ssha_anom


def save_files(date, output_dir, indicator_ds, globals_ds, pattern_and_anom_das):
    ds_and_paths = []

    fp_date = date.replace('-', '_')
    cycle_indicators_path = output_dir / 'cycle_indicators'
    cycle_indicators_path.mkdir(parents=True, exist_ok=True)
    indicator_output_path = cycle_indicators_path / f'{fp_date}_indicator.nc'
    ds_and_paths.append((indicator_ds, indicator_output_path))

    cycle_globals_path = output_dir / 'cycle_globals'
    cycle_globals_path.mkdir(parents=True, exist_ok=True)
    global_output_path = cycle_globals_path / f'{fp_date}_globals.nc'
    ds_and_paths.append((globals_ds, global_output_path))

    for pattern in pattern_and_anom_das.keys():
        pattern_anom_ds = pattern_and_anom_das[pattern]
        pattern_anom_ds = pattern_anom_ds.expand_dims(
            time=[pattern_anom_ds.time.values])

        cycle_pattern_anoms_path = output_dir / 'cycle_pattern_anoms' / pattern
        cycle_pattern_anoms_path.mkdir(parents=True, exist_ok=True)
        pattern_anoms_output_path = cycle_pattern_anoms_path / \
            f'{fp_date}_{pattern}_ssha_anoms.nc'
        ds_and_paths.append((pattern_anom_ds, pattern_anoms_output_path))

    encoding_each = {'zlib': True,
                     'complevel': 5,
                     'dtype': 'float32',
                     'shuffle': True,
                     '_FillValue': default_fillvals['f8']}

    for ds, path in ds_and_paths:

        coord_encoding = {}
        for coord in ds.coords:
            coord_encoding[coord] = {'_FillValue': None,
                                     'dtype': 'float32',
                                     'complevel': 6}

        var_encoding = {
            var: encoding_each for var in ds.data_vars}

        encoding = {**coord_encoding, **var_encoding}

        ds.to_netcdf(path, encoding=encoding)
        ds.close()

    return


def concat_files(indicator_dir, type, pattern=''):
    # Glob DAILY indicators
    daily_path = indicator_dir / f'DAILY/cycle_{type}s' / pattern
    daily_files = [x for x in daily_path.glob('*.nc') if x.is_file()]
    daily_files.sort()

    files = daily_files

    if pattern:
        print(f' - Reading {pattern} files')
    else:
        print(f' - Reading {type} files')

    all_ds = []
    for c in files:
        all_ds.append(xr.open_dataset(c))

    print(f' - Concatenating {type} files')
    concat_ds = xr.concat(all_ds, dim='time')
    all_ds = []

    return concat_ds


def indicators(output_path, reprocess):
    """
    This function calculates indicator values for each regridded cycle. Those are
    saved locally to avoid overloading memory. All locally saved indicator files 
    are combined into a single netcdf spanning the entire 1992 - NOW time period.
    """

    # Query for indicator doc on Solr
    fq = ['type_s:indicator']
    indicator_query = solr_utils.solr_query(fq)
    update = len(indicator_query) == 1

    if not update or reprocess:
        modified_time = '1992-01-01T00:00:00Z'
    else:
        indicator_metadata = indicator_query[0]
        modified_time = indicator_metadata['modified_time_dt']

        backup_dir = Path(f'{output_path}/indicator/backups')
        backup_dir.mkdir(parents=True, exist_ok=True)

        # Copy old indicator file as backup
        try:
            print('Making backup of existing indicator file.\n')
            indicator_path = indicator_metadata['indicator_filepath_s']
            old_time = str(indicator_metadata['modified_time_dt'])
            old_time = old_time.replace(':', '').replace('Z', '')

            backup_path = f'{backup_dir}/indicator_{old_time}.nc'
            copyfile(indicator_path, backup_path)

        except Exception as e:
            log.exception(f'Error creating indicator backup: {e}')

    # Query for update cycles after modified_time
    fq = ['type_s:gridded_cycle', 'processing_success_b:true',
          f'processing_time_dt:[{modified_time} TO NOW]']

    updated_cycles = solr_utils.solr_query(fq, sort='start_date_dt asc')

    # ONLY PROCEED IF THERE ARE CYCLES NEEDING CALCULATING
    if not updated_cycles:
        print('No regridded cycles modified since last index calculation.')
        return False

    time_format = "%Y-%m-%dT%H:%M:%S"

    chk_time = datetime.utcnow().strftime(time_format)

    print('Calculating new index values for modified cycles.\n')

    # ==============================================
    # Pattern preparation
    # ==============================================

    patterns = ['enso', 'pdo', 'iod']

    pattern_ds = dict()
    pattern_geo_bnds = dict()
    ann_cyc_in_pattern = dict()
    pattern_area_defs = dict()

    ref_dir = Path(f'ref_files/')

    # Global grid
    ecco_fname = 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc'
    ecco_latlon_grid = xr.open_dataset(ref_dir / ecco_fname)

    global_lon = ecco_latlon_grid.longitude.values
    global_lat = ecco_latlon_grid.latitude.values
    global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)

    pattern_area_defs['global'] = pr.geometry.SwathDefinition(lons=global_lon_m,
                                                              lats=global_lat_m)

    # load the monthly global sla climatology
    ann_ds = xr.open_dataset(ref_dir / 'ann_pattern.nc')

    # load patterns and select out the monthly climatology of sla variation
    # in each pattern
    for pattern in patterns:
        # load each pattern
        pattern_fname = pattern + '_pattern_and_index.nc'
        pattern_ds[pattern] = xr.open_dataset(ref_dir / pattern_fname)

        # get the geographic bounds of each sla pattern
        pattern_geo_bnds[pattern] = [float(pattern_ds[pattern].Latitude[0].values),
                                     float(
                                         pattern_ds[pattern].Latitude[-1].values),
                                     float(
                                         pattern_ds[pattern].Longitude[0].values),
                                     float(pattern_ds[pattern].Longitude[-1].values)]

        # extract the sla annual cycle in the region of each pattern
        ann_cyc_in_pattern[pattern] = ann_ds.sel(Latitude=slice(pattern_geo_bnds[pattern][0],
                                                                pattern_geo_bnds[pattern][1]),
                                                 Longitude=slice(pattern_geo_bnds[pattern][2],
                                                                 pattern_geo_bnds[pattern][3]))

        # Individual Patterns
        lon_m, lat_m = np.meshgrid(pattern_ds[pattern].Longitude.values,
                                   pattern_ds[pattern].Latitude.values)
        tmp_lon, tmp_lat = check_and_wrap(lon_m, lat_m)
        pattern_area_defs[pattern] = pr.geometry.SwathDefinition(lons=tmp_lon,
                                                                 lats=tmp_lat)

    # ==============================================
    # Calculate indicators for each updated (re)gridded cycle
    # ==============================================

    for cycle in updated_cycles:
        try:
            # Setup output directories
            output_dir = output_path / 'indicator' / 'daily'
            output_dir.mkdir(parents=True, exist_ok=True)

            cycle_ds = xr.open_dataset(cycle['filepath_s'])
            cycle_ds.close()

            date = cycle['date_dt'][:10]

            print(f' - Calculating index values for {date}')

            ct = np.datetime64(date)

            # Area mask the cycle data
            global_dam = cycle_ds.where(
                ecco_latlon_grid.maskC.isel(Z=0) > 0)['SSHA']
            global_dam = global_dam.where(global_dam)

            global_dam.name = 'SSHA_GLOBAL'
            global_dam.attrs['comment'] = 'Global SSHA land masked'
            global_dsm = global_dam.to_dataset()

            # Spatial Mean
            mean_da = calc_spatial_mean(global_dam, ecco_latlon_grid, ct)

            global_dam_removed_mean = global_dam - mean_da.values
            global_dam_removed_mean.attrs['comment'] = 'Global SSHA with global spatial mean removed'
            global_dsm['SSHA_GLOBAL_removed_global_spatial_mean'] = global_dam_removed_mean

            # Linear Trend
            trend = calc_linear_trend(ref_dir, cycle_ds)
            global_dsm['SSHA_GLOBAL_linear_trend'] = trend

            global_dam_detrended = global_dam - trend
            global_dam_detrended.attrs['comment'] = 'Global SSHA with linear trend removed'
            global_dsm['SSHA_GLOBAL_removed_linear_trend'] = global_dam_detrended

            if 'Z' in global_dsm.data_vars:
                global_dsm = global_dsm.drop_vars('Z')

            pattern_and_anom_das = {}

            all_indicators = []

            # Do the actual index calculation per pattern
            for pattern in patterns:
                pattern_lats = pattern_ds[pattern]['Latitude']
                pattern_lats = pattern_lats.rename({'Latitude': 'latitude'})
                pattern_lons = pattern_ds[pattern]['Longitude']
                pattern_lons = pattern_lons.rename({'Longitude': 'longitude'})
                pattern_lons, pattern_lats = check_and_wrap(pattern_lons,
                                                            pattern_lats)

                agg_da = global_dsm['SSHA_GLOBAL_removed_linear_trend'].sel(
                    longitude=pattern_lons, latitude=pattern_lats)
                agg_da.name = f'SSHA_{pattern}_removed_global_linear_trend'

                agg_ds = agg_da.to_dataset()
                agg_ds.attrs = cycle_ds.attrs

                index_calc, ct,  ssha_anom = calc_climate_index(agg_ds, pattern,
                                                                pattern_ds,
                                                                ann_cyc_in_pattern)

                anom_name = f'SSHA_{pattern}_removed_global_linear_trend_and_seasonal_cycle'
                ssha_anom.name = anom_name

                agg_ds[anom_name] = ssha_anom

                # Handle patterns and anoms
                pattern_and_anom_das[pattern] = agg_ds

                # Handle indicators and offsets
                indicator_da = xr.DataArray(index_calc[1], coords={'time': ct})
                indicator_da.name = f'{pattern}_index'
                all_indicators.append(indicator_da)

                offsets_da = xr.DataArray(index_calc[0], coords={'time': ct})
                offsets_da.name = f'{pattern}_offset'
                all_indicators.append(offsets_da)

            # Merge pattern indicators, offsets, and global spatial mean
            all_indicators.append(mean_da)
            indicator_ds = xr.merge(all_indicators)
            indicator_ds = indicator_ds.expand_dims(
                time=[indicator_ds.time.values])

            globals_ds = global_dsm
            globals_ds = globals_ds.expand_dims(time=[globals_ds.time.values])

            # Save indicators ds, global ds, and individual pattern ds for this one cycle
            save_files(date, output_dir, indicator_ds,
                       globals_ds, pattern_and_anom_das)

        except Exception as e:
            log.exception(e)

    print('\nCycle index calculation complete. ')
    print('Merging and saving final indicator products.\n')

    # ==============================================
    # Combine DAILY indicator files
    # ==============================================

    try:
        indicator_dir = output_path / 'indicator'

        # open_mfdataset is too slow so we glob instead
        indicators = concat_files(indicator_dir, 'indicator')
        print(' - Saving indicator file\n')
        indicators.to_netcdf(indicator_dir / 'indicators.nc')

        for pattern in patterns:
            pattern_anoms = concat_files(
                indicator_dir, 'pattern_anom', pattern)
            print(f' - Saving {pattern} anom file\n')
            pattern_anoms.to_netcdf(indicator_dir / f'{pattern}_anoms.nc')

            pattern_anoms = None

        globals_ds = concat_files(indicator_dir, 'global')
        print(' - Saving global file\n')
        globals_ds.to_netcdf(indicator_dir / 'globals.nc')

        globals_ds = None

    except Exception as e:
        log.exception(e)

    # ==============================================
    # Create or update indicator on Solr
    # ==============================================

    indicator_filepath = indicator_dir / 'indicators.nc'

    indicator_meta = {
        'type_s': 'indicator',
        'start_date_dt': np.datetime_as_string(indicators.time.values[0], unit='s'),
        'end_date_dt': np.datetime_as_string(indicators.time.values[-1], unit='s'),
        'modified_time_dt': chk_time,
        'indicator_filename_s': 'indicators.nc',
        'indicator_filepath_s': str(indicator_filepath),
        'indicator_checksum_s': file_utils.md5(indicator_filepath),
        'indicator_file_size_l': indicator_filepath.stat().st_size
    }

    if update:
        indicator_meta['prior_end_dt'] = indicator_query[0]['end_date_dt']
        indicator_meta['id'] = indicator_query[0]['id']

    # Update Solr with dataset metadata
    resp = solr_utils.solr_update([indicator_meta], r=True)

    if resp.status_code == 200:
        status = 'Successfully created or updated Solr index document'
        print(f'\n{status}')
    else:
        status = 'Failed to create or update Solr index document'
        print(f'\n{status}')

    return True
