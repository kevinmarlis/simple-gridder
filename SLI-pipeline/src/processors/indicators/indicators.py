"""

"""
import hashlib
import logging
import os
import pickle
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pyresample as pr
import requests
import xarray as xr
from netCDF4 import default_fillvals
from pyresample.kd_tree import resample_gauss
from pyresample.utils import check_and_wrap
from scipy.optimize import leastsq

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")


def md5(fname):
    """
    Creates md5 checksum from file
    """
    hash_md5 = hashlib.md5()
    with open(fname, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, fq, sort='date_dt asc'):
    """
    Queries Solr database using the filter query passed in.

    Params:
        config (dict): the dataset specific config file
        fq (List[str]): the list of filter query arguments

    Returns:
        response.json()['response']['docs'] (List[dict]): the Solr docs that satisfy the query
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    query_params = {'q': '*:*',
                    'fq': fq,
                    'rows': 300000,
                    'sort': sort}

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=query_params)
    return response.json()['response']['docs']


def solr_update(config, update_body):
    """
    Updates Solr database with list of docs. If a doc contains an existing id field,
    Solr will update or replace that existing doc with the new doc.

    Params:
        config (dict): the dataset specific config file
        update_body (List[dict]): the list of docs to update on Solr

    Returns:
        requests.post(url, json=update_body) (Response): the Response object from the post call
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    return requests.post(url, json=update_body)


def calc_climate_index(agg_ds, pattern, pattern_ds, ann_cyc_in_pattern, method=3):
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

    if method == 1:
        # Method 1, use scipy's least squares:
        # some kind of first guess
        params = [0, 0]

        # minimizes func1
        result = leastsq(func1, params, (pattern_to_fit, ssha_anom_to_fit))
        offset = result[0][0]
        index = result[0][1]

        # now minimize against ssha_to_fit
        params = [0, 0]
        # minimizes func1
        result = leastsq(func1, params, (pattern_to_fit, ssha_to_fit))

        offset_b = result[0][0]
        index_b = result[0][1]

    elif method == 2:
        # Method 2, like a boss
        # B_hat = inv(X.TX)X.T Y

        # constrct design matrix
        # X: [1 x_0]
        #    [1 x_1]
        #    [1 x_2]
        #    ....
        #   [ 1 x_n-1]

        # to minimize J = (y_e - y)**2
        #     where y_e = b_0 * 1 + b_1 * x
        #
        # B_hat = [b_0]
        #         [b_1]

        # design matrix
        X = np.vstack((np.ones(len(pattern_to_fit)), pattern_to_fit)).T

        # Good old Gauss
        B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T),
                          ssha_anom_to_fit.T)
        offset = B_hat[0]
        index = B_hat[1]

        # now minimize ssha_to_fit
        B_hat = np.matmul(np.matmul(np.linalg.inv(
            np.matmul(X.T, X)), X.T), ssha_to_fit.T)
        offset_b = B_hat[0]
        index_b = B_hat[1]

    elif method == 3:
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


# one of the ways of doing the LS fit is with the scipy.optimize leastsq
# requires defining a function with the following syntax
def func1(params, x, y):
    m, b = params[1], params[0]
    # the expression connecting y and the parameters is arbitrary
    # but here we just do the old standby
    residual = y - (m*x + b)
    return residual


def indicators(config, output_path, reprocess, log_time):
    """
    """

    # Set file handler for log using output_path
    formatter = logging.Formatter('%(asctime)s:  %(message)s')

    logs_path = Path(output_path / f'logs/{log_time}')
    logs_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(logs_path / 'indicator.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    log.addHandler(file_handler)

    # Import ecco access tools
    generalized_functions_path = Path(
        f'{Path(__file__).resolve().parents[4]}/SLI-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    method = 3
    output_dir = output_path / 'indicator' / f'ben_trend_method{method}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Query for indicator doc on Solr
    fq = ['type_s:indicator']
    indicator_query = solr_query(config, fq)
    update = len(indicator_query) == 1

    if not update or reprocess:
        modified_time = '1992-01-01T00:00:00Z'
    else:
        indicator_metadata = indicator_query[0]
        modified_time = indicator_metadata['modified_time_dt']

    # Query for update cycles after modified_time
    # TODO: Modify to select gridded data when available
    fq = ['type_s:regridded_cycle', 'processing_success_b:true',
          f'processing_time_dt:[{modified_time} TO NOW]']

    # TODO: this is for development
    # fq = ['(type_s:regridded_cycle AND original_data_type_s:along_track AND processing_success_b:true)']
    # fq = ['(type_s:regridded_cycle AND original_data_type_s:along_track AND processing_success_b:true) OR \
    #         (type_s:regridded_cycle AND original_data_type_s:gridded AND \
    #         start_date_dt:[1992-01-01T00:00:00Z TO 2017-01-01T00:00:00Z] AND processing_success_b:true)']

    updated_cycles = solr_query(config, fq, sort='start_date_dt asc')

    # ONLY PROCEED IF THERE ARE CYCLES NEEDING CALCULATING
    if not updated_cycles:
        print('No regridded cycles modified since last index calculation.')
        return

    time_format = "%Y-%m-%dT%H:%M:%S"

    chk_time = datetime.utcnow().strftime(time_format)

    print('Calculating new index values for modified cycles.\n')

    # PRELIMINARY STUFF, CAN BE DONE BEFORE CALLING THE ROUTINE TO DO THE INDEXING
    # ----------------------------------------------------------------------------
    # load patterns and select out the monthly climatology of sla variation
    # in each pattern
    patterns = ['enso', 'pdo', 'iod']

    # ben's patterns (in NetCDF form from his original matlab format)
    bh_dir = Path().resolve() / 'SLI-pipeline' / 'ref_grds'

    # PREPARE THE PATTERNS

    # load the monthly global sla climatology
    ann_ds = xr.open_dataset(bh_dir / 'ann_pattern.nc')

    pattern_ds = dict()
    pattern_geo_bnds = dict()
    ann_cyc_in_pattern = dict()
    pattern_area_defs = dict()

    for pattern in patterns:
        # load each pattern
        pattern_fname = pattern + '_pattern_and_index.nc'
        pattern_ds[pattern] = xr.open_dataset(bh_dir / pattern_fname)

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
        pattern_area_defs[pattern] = pr.geometry.SwathDefinition(
            lons=tmp_lon, lats=tmp_lat)

    # Annual Cycle
    lon_m, lat_m = np.meshgrid(
        ann_ds.Longitude.values, ann_ds.Latitude.values)
    tmp_lon, tmp_lat = check_and_wrap(lon_m, lat_m)
    ann_area_def = pr.geometry.SwathDefinition(lons=tmp_lon, lats=tmp_lat)

    # Global
    ecco_fname = 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc'
    ecco_latlon_grid = xr.open_dataset(bh_dir / ecco_fname)

    global_lon = ecco_latlon_grid.longitude.values
    global_lat = ecco_latlon_grid.latitude.values
    global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)

    pattern_area_defs['global'] = pr.geometry.SwathDefinition(lons=global_lon_m,
                                                              lats=global_lat_m)

    ############################
    # THE MAIN LOOP
    ############################

    for cycle in updated_cycles:
        cycle_ds = xr.open_dataset(cycle['filepath_s'])
        cycle_ds.close()

        cycle_type = cycle['original_data_type_s']
        date = cycle['center_date_dt'][:10]

        print(f' - Calculating index values for {date}')

        ct = np.datetime64(cycle_ds.cycle_center)

        global_dam = cycle_ds.where(
            ecco_latlon_grid.maskC.isel(Z=0) > 0)['SSHA']
        global_dam.name = 'SSHA_GLOBAL'
        global_dam.attrs['comment'] = 'Global SSHA land masked'
        global_dsm = global_dam.to_dataset()

        # Spatial Mean
        nzp = np.where(~np.isnan(global_dam), 1, np.nan)
        area_nzp = np.sum(nzp * ecco_latlon_grid.area)
        spatial_mean = float(
            np.nansum(global_dam * ecco_latlon_grid.area) / area_nzp)
        mean_da = xr.DataArray(spatial_mean, coords={
            'time': ct}, attrs=global_dam.attrs)
        mean_da.name = 'spatial_mean'
        mean_da.attrs['comment'] = 'Global SSHA spatial mean'

        global_dam_removed_mean = global_dam - spatial_mean
        global_dam_removed_mean.attrs['comment'] = 'Global SSHA with global spatial mean removed'
        global_dsm['SSHA_GLOBAL_removed_global_spatial_mean'] = global_dam_removed_mean

        # Linear Trend
        # trend_ds = xr.open_dataset(bh_dir / 'pointwise_sealevel_trend.nc')
        ben_trend_ds = xr.open_dataset(
            bh_dir / 'BH_offset_and_trend_v0_new_grid.nc')

        # time_diff = cycle_time - initial time (1992-10-02) converted to seconds
        # cycle_time - initial time gets count of days
        time_diff = (cycle_ds.time.values.astype(
            'datetime64[D]') - np.datetime64('1992-10-02')).astype(np.int32) * 24 * 60 * 60

        # trend = time_diff * trend_ds['pointwise_sealevel_trend']
        ben_trend = time_diff * \
            ben_trend_ds['BH_sea_level_trend_meters_per_second'] + \
            ben_trend_ds['BH_sea_level_offset_meters']

        global_dsm['SSHA_GLOBAL_linear_ben_trend'] = ben_trend

        # global_dam_detrended = global_dam - trend
        global_dam_detrended = global_dam - ben_trend

        global_dam_detrended.attrs = {
            'comment': 'Global SSHA with linear trend removed'}
        global_dsm['SSHA_GLOBAL_removed_linear_trend'] = global_dam_detrended

        if 'Z' in global_dsm.data_vars:
            global_dsm = global_dsm.drop_vars('Z')

        indicators_agg_das = {}
        offset_agg_das = {}
        pattern_and_anom_das = {}

        for pattern in patterns:
            pattern_lats = pattern_ds[pattern]['Latitude']
            pattern_lats = pattern_lats.rename({'Latitude': 'latitude'})
            pattern_lons = pattern_ds[pattern]['Longitude']
            pattern_lons = pattern_lons.rename({'Longitude': 'longitude'})
            pattern_lons, pattern_lats = check_and_wrap(pattern_lons,
                                                        pattern_lats)

            # Calculate pattern mean sea level
            # pattern_area_dam = global_dam.sel(longitude=pattern_lons, latitude=pattern_lats)
            # pattern_nzp = np.where(~np.isnan(pattern_area_dam), 1, np.nan)
            # pattern_ecco_latlon_grid = ecco_latlon_grid.sel(longitude=pattern_lons,
            #                                                 latitude=pattern_lats)
            # pattern_area_nzp = np.sum(pattern_nzp * pattern_ecco_latlon_grid.area)

            # pattern_area_spatial_mean = float(
            #     np.nansum(pattern_area_dam * pattern_ecco_latlon_grid.area) / pattern_area_nzp)

            agg_da = global_dsm['SSHA_GLOBAL_removed_linear_trend'].sel(
                longitude=pattern_lons, latitude=pattern_lats)
            agg_da.name = f'SSHA_{pattern}_removed_global_linear_trend'

            # agg_da.attrs =
            agg_ds = agg_da.to_dataset()
            agg_ds.attrs = cycle_ds.attrs

            index_calc, ct,  ssha_anom = calc_climate_index(agg_ds, pattern, pattern_ds,
                                                            ann_cyc_in_pattern,
                                                            method=method)

            anom_name = f'SSHA_{pattern}_removed_global_linear_trend_and_seasonal_cycle'
            ssha_anom.name = anom_name

            agg_ds[anom_name] = ssha_anom

            pattern_and_anom_das[pattern] = agg_ds

            indicator_da = xr.DataArray(index_calc[1], coords={'time': ct})
            indicator_da.name = f'{pattern}_index'
            indicators_agg_das[pattern] = indicator_da

            offsets_da = xr.DataArray(index_calc[0], coords={'time': ct})
            offsets_da.name = f'{pattern}_offset'
            offset_agg_das[pattern] = offsets_da

        # Concatenate the list of individual DAs along time
        # Merge into a single DataSet and append that pattern to all_indicators list
        # List to hold DataSet objects for each pattern
        all_indicators = []

        for pattern in patterns:
            all_indicators.append(
                xr.merge([offset_agg_das[pattern], indicators_agg_das[pattern], mean_da]))

        # FINISHED THROUGH ALL PATTERNS
        # append all the datasets together into a single dataset to rule them all
        indicator_ds = xr.merge(all_indicators)
        indicator_ds = indicator_ds.expand_dims(
            time=[indicator_ds.time.values])

        globals_ds = global_dsm
        globals_ds = globals_ds.expand_dims(time=[globals_ds.time.values])

        all_ds = []

        fp_date = date.replace('-', '_')

        cycle_indicators_path = output_dir / 'cycle_indicators'
        cycle_indicators_path.mkdir(parents=True, exist_ok=True)
        indicator_output_path = cycle_indicators_path / \
            f'{fp_date}_indicator.nc'
        all_ds.append((indicator_ds, indicator_output_path))

        cycle_globals_path = output_dir / 'cycle_globals'
        cycle_globals_path.mkdir(parents=True, exist_ok=True)
        global_output_path = cycle_globals_path / f'{fp_date}_globals.nc'
        all_ds.append((globals_ds, global_output_path))

        for pattern in patterns:
            pattern_anom_ds = pattern_and_anom_das[pattern]
            pattern_anom_ds = pattern_anom_ds.expand_dims(
                time=[pattern_anom_ds.time.values])

            cycle_pattern_anoms_path = output_dir / 'cycle_pattern_anoms' / pattern
            cycle_pattern_anoms_path.mkdir(parents=True, exist_ok=True)
            pattern_anoms_output_path = cycle_pattern_anoms_path / \
                f'{fp_date}_{pattern}_ssha_anoms.nc'

            all_ds.append((pattern_anom_ds, pattern_anoms_output_path))

        # # NetCDF encoding
        encoding_each = {'zlib': True,
                         'complevel': 5,
                         'dtype': 'float32',
                         'shuffle': True,
                         '_FillValue': default_fillvals['f8']}

        for ds, output_path in all_ds:

            coord_encoding = {}
            for coord in ds.coords:
                coord_encoding[coord] = {'_FillValue': None,
                                         'dtype': 'float32',
                                         'complevel': 6}

                # if 'Time' in coord:
                #     coord_encoding[coord] = {'_FillValue': None,
                #                              'zlib': True,
                #                              'contiguous': False,
                #                              'shuffle': False}

            var_encoding = {
                var: encoding_each for var in ds.data_vars}

            encoding = {**coord_encoding, **var_encoding}

            ds.to_netcdf(output_path, encoding=encoding)
            ds.close()

    print('\nCycle index calculation complete. ')
    print('Merging and saving final indicator products.\n')

    # open_mfdataset is too slow so we glob instead
    print(' - Reading indicator files')
    cycle_indicators_path = output_dir / 'cycle_indicators'
    files = [x for x in cycle_indicators_path.glob('*.nc') if x.is_file()]
    files.sort()

    ind_ds = []
    for c in files:
        ind_ds.append(xr.open_dataset(c))

    print(' - Concatenating indicator files')
    indicators = xr.concat(ind_ds, dim='time')
    ind_ds = []
    print(' - Saving indicator file')
    indicators.to_netcdf(output_dir / 'indicators.nc')

    indicators = None

    for pattern in patterns:
        print(f'\n - Reading {pattern} anom files')
        cycle_pattern_anom_path = output_dir / 'cycle_pattern_anoms' / pattern

        files = [x for x in cycle_pattern_anom_path.glob(
            '*.nc') if x.is_file()]
        files.sort()
        pa_ds = []
        for c in files:
            pa_ds.append(xr.open_dataset(c))

        print(f' - Concatenating {pattern} anom files')
        pattern_anoms = xr.concat(pa_ds, dim='time')
        pa_ds = []
        print(f' - Saving {pattern} anom file')
        pattern_anoms.to_netcdf(output_dir / f'{pattern}_anoms.nc')

        pattern_anoms = None

    print('\n - Reading global files')
    cycle_global_path = output_dir / 'cycle_globals'

    files = [x for x in cycle_global_path.glob('*.nc') if x.is_file()]
    files.sort()
    g_ds = []
    for c in files:
        g_ds.append(xr.open_dataset(c))

    print(' - Concatenating global files')
    global_ds = xr.concat(g_ds, dim='time')
    g_ds = []
    print(' - Saving global file\n')
    global_ds.to_netcdf(output_dir / 'globals.nc')

    global_ds = None

    # ==============================================
    # Create or update indicator on Solr
    # ==============================================

    # indicator_meta = {
    #     'type_s': 'indicator',
    #     'start_date_dt': np.datetime_as_string(indicator_ds.time.values[0], unit='s'),
    #     'end_date_dt': np.datetime_as_string(indicator_ds.time.values[-1], unit='s'),
    #     'modified_time_dt': chk_time,
    #     'indicator_filename_s': indicator_filename,
    #     'indicator_filepath_s': str(indicator_output_path),
    #     'indicator_checksum_s': md5(indicator_output_path),
    #     'indicator_file_size_l': indicator_output_path.stat().st_size,
    #     'patterns_anoms_filename_s': patterns_anoms_filename,
    #     'patterns_anoms_filepath_s': str(patterns_and_anoms_output_path),
    #     'patterns_anoms_checksum_s': md5(patterns_and_anoms_output_path),
    #     'patterns_anoms_file_size_l': patterns_and_anoms_output_path.stat().st_size,
    #     'globals_filename_s': global_filename,
    #     'globals_filepath_s': str(global_output_path),
    #     'globals_checksum_s': md5(global_output_path),
    #     'globals_file_size_l': global_output_path.stat().st_size
    # }

    # if update:
    #     indicator_meta['id'] = indicator_query[0]['id']

    # # Update Solr with dataset metadata
    # resp = solr_update(config, [indicator_meta])

    # if resp.status_code == 200:
    #     print('\nSuccessfully created or updated Solr index document')
    # else:
    #     print('\nFailed to create or update Solr index document')
