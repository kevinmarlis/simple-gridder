"""

"""
import os
import sys
import hashlib
import logging
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
import pyresample as pr
import requests
import xarray as xr
import xesmf as xe
# from netCDF4 import default_fillvals
from pyresample.utils import check_and_wrap
from scipy.optimize import leastsq
import cartopy.crs as crs
import cartopy.feature as cfeature
from pyresample.kd_tree import resample_gauss

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class HiddenPrints:
    """
    Temporarily hides print statements, especially useful for library functions.

    ex:
    with HiddenPrints():
        foo(bar)

    Print statements called by foo will be intercepted.
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


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


def calc_grid_climate_index(agg_ds, pattern, pattern_ds, ann_cyc_in_pattern, weights_dir):
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

    method = 2

    # TODO: this could be problematic if the first case is not a np.datetime64 object
    # not sure if second case is needed (could be from an early cycle draft)
    # extract center time of this agg field
    if 'cycle_center' in agg_ds.attrs:
        center_time = agg_ds.Time[0].values

    elif 'time_center' in agg_ds.attrs:
        center_time = np.datetime64(agg_ds.time_center)

    # determine its month
    agg_ds_center_mon = int(str(center_time)[5:7])

    pattern_field = pattern_ds[pattern][f'{pattern}_pattern'].values

    ds_in = (agg_ds['SSHA'].rename({'Longitude': 'lon', 'Latitude': 'lat'}).isel(Time=0)).T
    ds_out = pattern_ds[pattern][f'{pattern}_pattern'].rename(
        {'Longitude': 'lon', 'Latitude': 'lat'})

    weight_fp = f'{weights_dir}/1812_to_{pattern}.nc'
    if not os.path.exists(weight_fp):
        print(f'Creating {pattern} weight file.')

    # HiddenPrints class keeps xesmf.Regridder from printing details about weights
    with HiddenPrints():
        regridder = xe.Regridder(ds_in, ds_out, 'bilinear', filename=weight_fp, reuse_weights=True)

    ssha_to_pattern_da = regridder(ds_in)

    ssha_to_pattern_da = ssha_to_pattern_da.assign_coords(coords={'time': center_time})

    # remove the monthly mean pattern from the gridded ssha
    # now ssha_anom is w.r.t. seasonal cycle and MDT
    ssha_anom = ssha_to_pattern_da.values - \
        ann_cyc_in_pattern[pattern].ann_pattern.sel(month=agg_ds_center_mon).values/1e3

    # set ssha_anom to nan wherever the original pattern is nan
    ssha_anom = np.where(~np.isnan(ds_out), ssha_anom, np.nan)

    # extract out all non-nan values of ssha_anom, these are going to
    # be the points that we fit
    nonnans = ~np.isnan(ssha_anom)
    ssha_anom_to_fit = ssha_anom[nonnans]

    # do the same for the pattern
    pattern_to_fit = pattern_field[nonnans]/1e3

    # just for fun extract out same points from ssha, we'll see if
    # removing the monthly climatology makes much of a difference
    ssha_to_fit = ssha_to_pattern_da.copy(deep=True)
    ssha_to_fit = ssha_to_pattern_da.values[nonnans]

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
        B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), ssha_to_fit.T)
        offset_b = B_hat[0]
        index_b = B_hat[1]

    LS_result = [offset, index, offset_b, index_b]

    return LS_result, center_time


def calc_at_climate_index(agg_ds,
                          pattern,
                          pattern_ds,
                          ann_cyc_in_pattern,
                          method=2):

    LS_result = []

    # extract center time of this agg field
    if 'cycle_center' in agg_ds.attrs:
        ct = np.datetime64(agg_ds.cycle_center)

    elif 'time_center' in agg_ds.attrs:
        ct = np.datetime64(agg_ds.time_center)

    # determine its month
    agg_ds_center_mon = int(str(ct)[5:7])
    print(ct, agg_ds_center_mon)

    pattern_field = pattern_ds[pattern][f'{pattern}_pattern']

    ssha_anom = agg_ds[f'SSHA_{pattern}_removed_global_spatial_mean'] - \
        ann_cyc_in_pattern[pattern].ann_pattern.sel(month=agg_ds_center_mon)/1e3

    # set ssha_anom to nan wherever the original pattern is nan
    # ssha_anom = np.where(~np.isnan(pattern_field), ssha_anom, np.nan)
    ssha_anom = ssha_anom.where(~np.isnan(pattern_field))

    # extract out all non-nan values of ssha_anom, these are going to
    # be the points that we fit
    nn = ~np.isnan(ssha_anom.values)
    ssha_anom_to_fit = ssha_anom.values[nn]

    # do the same for the pattern and also change dimension to meters
    pattern_to_fit = pattern_field.values[nn]/1e3

    # ssha without removing the anomaly
    ssha_to_fit = agg_ds.copy(deep=True)
    ssha_to_fit = agg_ds[f'SSHA_{pattern}_removed_global_spatial_mean'].values[nn]

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
        B_hat = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T),
                          ssha_to_fit.T)
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

    return LS_result, ct, ssha_anom


# one of the ways of doing the LS fit is with the scipy.optimize leastsq
# requires defining a function with the following syntax
def func1(params, x, y):
    m, b = params[1], params[0]
    # the expression connecting y and the parameters is arbitrary
    # but here we just do the old standby
    residual = y - (m*x + b)
    return residual


def interp_ssha_points_to_pattern(pattern_area_def,
                                  ssha, ssha_lon, ssha_lat,
                                  roi=1e5,
                                  sigma=25000,
                                  neighbours=500):

    # Define the 'swath' as the lats/lon pairs of the model grid

    ssha_lat_nn = ssha_lat[~np.isnan(ssha)]
    ssha_lon_nn = ssha_lon[~np.isnan(ssha)]
    ssha_nn = ssha[~np.isnan(ssha)]

    if np.sum(~np.isnan(ssha_nn)) > 0:
        tmp_ssha_lons, tmp_ssha_lats = check_and_wrap(ssha_lon_nn.ravel(),
                                                      ssha_lat_nn.ravel())

        ssha_grid = pr.geometry.SwathDefinition(lons=tmp_ssha_lons, lats=tmp_ssha_lats)

        ssha_pts_to_pattern = resample_gauss(ssha_grid, ssha_nn,
                                             pattern_area_def,
                                             radius_of_influence=roi,
                                             sigmas=sigma,
                                             fill_value=np.NaN, neighbours=neighbours)

    else:
        print('--- entire cycle is NaN')
        ssha_pts_to_pattern = None

    return ssha_lon_nn, ssha_lat_nn, ssha_nn, ssha_pts_to_pattern


def indicators(config, output_path, reprocess=False):
    """
    Here
    """

    indicator_filename = 'indicator.nc'
    patterns_anoms_filename = 'patterns_and_ssha_anoms.nc'
    global_filename = 'globals.nc'
    output_dir = output_path / 'indicator'
    output_dir.mkdir(parents=True, exist_ok=True)
    indicator_output_path = output_dir / indicator_filename
    patterns_and_anoms_output_path = output_dir / patterns_anoms_filename
    global_output_path = output_dir / global_filename

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
    fq = ['type_s:cycle', 'processing_success_b:true',
          f'processing_time_dt:[{modified_time} TO NOW]']
    updated_cycles = solr_query(config, fq, sort='start_date_dt asc')

    # ONLY PROCEED IF THERE ARE CYCLES NEEDING CALCULATING
    if not updated_cycles:
        print('No cycles modified since last index calculation.')
        return

    # Group together along track cycles with the same dates
    grid_cycles = [cycle for cycle in updated_cycles if cycle['index_type_s'] == 'gridded']
    grid_cycle_starts = {cycle['start_date_dt'] for cycle in grid_cycles}

    at_cycles = [cycle for cycle in updated_cycles if cycle['index_type_s'] == 'along_track']
    at_cycle_starts = {cycle['start_date_dt'] for cycle in at_cycles}
    at_cycle_starts = sorted(at_cycle_starts)

    time_format = "%Y-%m-%dT%H:%M:%S"

    chk_time = datetime.utcnow().strftime(time_format)

    print('Calculating new index values for modified cycles.\n')

    # PRELIMINARY STUFF, CAN BE DONE BEFORE CALLING THE ROUTINE TO DO THE INDEXING
    # ----------------------------------------------------------------------------
    # load patterns and select out the monthly climatology of sla variation
    # in each pattern
    patterns = ['enso', 'pdo', 'iod']

    # ben's patterns (in NetCDF form from his original matlab format)
    bh_dir = Path().resolve() / 'Sea-Level-Indicators' / 'SLI-pipeline' / 'ref_grds'

    # weights dir is used to store remapping weights for xemsf regrid operation
    weights_dir = output_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)

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
                                     float(pattern_ds[pattern].Latitude[-1].values),
                                     float(pattern_ds[pattern].Longitude[0].values),
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
        pattern_area_defs[pattern] = pr.geometry.SwathDefinition(lons=tmp_lon, lats=tmp_lat)

    # Annual Cycle
    lon_m, lat_m = np.meshgrid(ann_ds.Longitude.values, ann_ds.Latitude.values)
    tmp_lon, tmp_lat = check_and_wrap(lon_m, lat_m)
    ann_area_def = pr.geometry.SwathDefinition(lons=tmp_lon, lats=tmp_lat)

    # Global
    ecco_fname = 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc'
    ecco_latlon_grid = xr.open_dataset(bh_dir / ecco_fname)

    global_lon = ecco_latlon_grid.longitude.values
    global_lat = ecco_latlon_grid.latitude.values
    global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)

    pattern_area_defs['global'] = pr.geometry.SwathDefinition(lons=global_lon_m, lats=global_lat_m)

    ############################
    # THE MAIN LOOP
    ############################

    # List to hold DataSet objects for each pattern
    all_indicators = []
    all_patterns_and_anoms = []
    all_globals = []

    # key (pattern) : value (list of DAs with index at a single time)
    indicators_agg_das = defaultdict(list)
    spatial_mean_das = []
    offset_agg_das = defaultdict(list)
    pattern_and_anom_das = defaultdict(list)
    global_dss = []

    # for cycle in grid_cycles:
    #     agg_file = cycle['filepath_s']
    #     cycle_type = cycle['index_type_s']
    #     date = agg_file[-18:-10]
    #     date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'

    #     print(f' - Calculating index values for {date}')

    #     cycle_ds = xr.open_dataset(agg_file)
    #     cycle_ds.close()

    #     indicators_agg_da = None
    #     offsets_agg_da = None

    #     for pattern in patterns:

    #         index_calc, ct = calc_grid_climate_index(cycle_ds, pattern, pattern_ds,
    #                                                  ann_cyc_in_pattern, weights_dir)
    #         # ssha_anom.name = f'{pattern}_SSHA_anomaly'
    #         # ssha_anom_das[pattern].append(ssha_anom)

    #         # create a DataArray Object with a single scalar value, the index
    #         # for this pattern at this one time.
    #         indicator_da = xr.DataArray(index_calc[1], coords={'time': ct})
    #         indicator_da.name = f'{pattern}_index'
    #         indicators_agg_das[pattern].append(indicator_da)

    #         # create a DataArray Object with a single scalar value, the
    #         # the offset of the climate indices (the constant term in the
    #         # least squares fit, not really interesting but maybe good to have)
    #         offsets_da = xr.DataArray(index_calc[0], coords={'time': ct})
    #         offsets_da.name = f'{pattern}_offset'
    #         offset_agg_das[pattern].append(offsets_da)

    # Loop through unique cycle dates
    for cycle_start in at_cycle_starts:

        # Get all along track cycles with this date
        fq = ['type_s:cycle', 'processing_success_b:true', 'index_type_s:along_track',
              f'start_date_dt:[{cycle_start} TO {cycle_start}]']
        same_period_cycles = solr_query(config, fq)

        # Concat and sort all along track cycles with the same cycle period
        ats = []
        for cycle in same_period_cycles:
            ds = xr.open_dataset(cycle['filepath_s'])
            ats.append(ds)

        at_ds = xr.concat(ats, 'Time')
        at_ds = at_ds.sortby('Time')

        if 'cycle_center' in at_ds.attrs:
            ct = np.datetime64(at_ds.cycle_center)

        elif 'time_center' in at_ds.attrs:
            ct = np.datetime64(at_ds.time_center)

        # Prepare global versions
        ssha_lon_nn, ssha_lat_nn, sha_nn, ssha_pts_to_pattern = \
            interp_ssha_points_to_pattern(pattern_area_defs['global'],
                                          at_ds.SSHA.values.ravel(),
                                          at_ds.SSHA.Longitude.values.ravel(),
                                          at_ds.SSHA.Latitude.values.ravel(),
                                          roi=4e5, sigma=1.5e5, neighbours=5)

        global_da = xr.DataArray(ssha_pts_to_pattern, dims=['latitude', 'longitude'],
                                 coords={'longitude': global_lon,
                                         'latitude': global_lat})
        global_da = global_da.assign_coords(coords={'Time': np.datetime64(at_ds.cycle_center)})

        global_dam = global_da.where(ecco_latlon_grid.maskC.isel(Z=0) > 0)
        nzp = np.where(~np.isnan(global_dam), 1, np.nan)
        area_nzp = np.sum(nzp * ecco_latlon_grid.area)

        spatial_mean = float(np.nansum(global_dam * ecco_latlon_grid.area) / area_nzp)
        mean_da = xr.DataArray(spatial_mean, coords={'time': ct})
        mean_da.name = 'spatial_mean'
        spatial_mean_das.append(mean_da)

        global_dam_removed_mean = global_dam - spatial_mean

        # Make dataset with global_da and global_dam_removed_mean
        global_dam.name = 'SSHA_GLOBAL'
        global_ds = global_dam.to_dataset()
        global_ds['SSHA_GLOBAL_removed_global_spatial_mean'] = global_dam_removed_mean
        global_ds = global_ds.drop_vars('Z')

        global_dss.append(global_ds)

        cycle_type = 'along_track'
        date = same_period_cycles[0]['filepath_s'][-18:-10]
        date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'
        method = 2

        print(f' - Calculating index values for {date}')

        indicators_agg_da = None
        offsets_agg_da = None

        for pattern in patterns:

            pattern_lats = pattern_ds[pattern]['Latitude']
            pattern_lons = pattern_ds[pattern]['Longitude']
            pattern_lons, pattern_lats = check_and_wrap(pattern_lons, pattern_lats)

            agg_da = global_dam_removed_mean.sel(longitude=pattern_lons, latitude=pattern_lats)
            agg_da.name = f'SSHA_{pattern}_removed_global_spatial_mean'
            agg_da = agg_da.assign_coords(coords={'Time': np.datetime64(at_ds.cycle_center)})

            agg_ds = agg_da.to_dataset()
            agg_ds.attrs = at_ds.attrs

            index_calc, ct,  ssha_anom = calc_at_climate_index(agg_ds,
                                                               pattern,
                                                               pattern_ds,
                                                               ann_cyc_in_pattern,
                                                               method=method)

            anom_name = f'SSHA_{pattern}_removed_global_spatial_mean_and_seasonal_cycle'
            ssha_anom.name = anom_name

            agg_ds[anom_name] = ssha_anom
            agg_ds = agg_ds.drop_vars(['Z', 'latitude', 'longitude'])

            pattern_and_anom_das[pattern].append(agg_ds)

            # create a DataArray Object with a single scalar value, the index
            # for this pattern at this one time.
            indicator_da = xr.DataArray(index_calc[1], coords={'time': ct})
            indicator_da.name = f'{pattern}_index'
            indicators_agg_das[pattern].append(indicator_da)

            # create a DataArray Object with a single scalar value, the
            # the offset of the climate indices (the constant term in the
            # least squares fit, not really interesting but maybe good to have)
            offsets_da = xr.DataArray(index_calc[0], coords={'time': ct})
            offsets_da.name = f'{pattern}_offset'
            offset_agg_das[pattern].append(offsets_da)

    all_globals.append(xr.concat(global_dss, dim='Time'))

    # Concatenate the list of individual DAs along time
    # Merge into a single DataSet and append that pattern to all_indicators list
    for pattern in patterns:
        indicators_agg_da = xr.concat(indicators_agg_das[pattern], 'time')
        offsets_agg_da = xr.concat(offset_agg_das[pattern], 'time')
        spatial_mean_da = xr.concat(spatial_mean_das, 'time')
        all_indicators.append(xr.merge([offsets_agg_da, indicators_agg_da, spatial_mean_da]))

        pattern_and_anom_da = xr.concat(pattern_and_anom_das[pattern], 'Time')
        all_patterns_and_anoms.append(pattern_and_anom_da)

    # FINISHED THROUGH ALL PATTERNS
    # append all the datasets together into a single dataset to rule them all
    new_indicators = xr.merge(all_indicators)
    new_patterns_and_anoms = xr.merge(all_patterns_and_anoms)
    new_globals = xr.merge(all_globals)

    # Open existing indicator ds to add new values if needed
    if update:
        print('\nAdding calculated values to indicator netCDF.')

        # PROCESS INDICATORS
        indicator_ds = xr.open_dataset(indicator_metadata['indicator_filepath_s'])

        # Remove times of new indicator values from original indicator DS if they exist
        # (this effectively updates the values)
        print(1)
        indicator_ds = indicator_ds.where(~indicator_ds['time'].isin(
            np.unique(new_indicators['time'])), drop=True)
        print(2)

        # And use xr.concat to add in the new values (concat will create multiple entries for the
        # same time value).
        indicator_ds = xr.concat([indicator_ds, new_indicators], 'time')
        print(3)

        # Finally, sort to get things in the right order
        indicator_ds = indicator_ds.sortby('time')
        print(4)

        # Reapeat for other files
        patterns_and_anoms_ds = xr.open_dataset(indicator_metadata['patterns_anoms_filepath_s'])
        patterns_and_anoms_ds.load()
        patterns_and_anoms_ds = patterns_and_anoms_ds.where(~patterns_and_anoms_ds['Time'].isin(
            np.unique(new_patterns_and_anoms['Time'])), drop=True)
        patterns_and_anoms_ds = xr.concat([patterns_and_anoms_ds, new_patterns_and_anoms], 'Time')
        patterns_and_anoms_ds = patterns_and_anoms_ds.sortby('Time')

        # PROCESS GLOBALS

        globals_ds = xr.open_dataset(indicator_metadata['globals_filepath_s'])
        globals_ds.load()
        globals_ds = globals_ds.where(~globals_ds['Time'].isin(
            np.unique(new_globals['Time'])), drop=True)
        globals_ds = xr.concat([globals_ds, new_globals], 'Time')
        globals_ds = globals_ds.sortby('Time')

    else:
        indicator_ds = new_indicators
        patterns_and_anoms_ds = new_patterns_and_anoms
        globals_ds = new_globals

    # # NetCDF encoding
    # encoding_each = {'zlib': True,
    #                  'complevel': 5,
    #                  'dtype': 'float32',
    #                  'shuffle': True,
    #                  '_FillValue': default_fillvals['f8']}

    # coord_encoding = {}
    # for coord in indicator_ds.coords:
    #     coord_encoding[coord] = {'_FillValue': None,
    #                              'dtype': 'float32',
    #                              'complevel': 6}

    #     if 'Time' in coord:
    #         coord_encoding[coord] = {'_FillValue': None,
    #                                  'zlib': True,
    #                                  'contiguous': False,
    #                                  'shuffle': False}

    # var_encoding = {var: encoding_each for var in indicator_ds.data_vars}

    # encoding = {**coord_encoding, **var_encoding}

    indicator_ds.to_netcdf(indicator_output_path)
    patterns_and_anoms_ds.to_netcdf(patterns_and_anoms_output_path)
    globals_ds.to_netcdf(global_output_path)

    # ==============================================
    # Create or update indicator on Solr
    # ==============================================

    indicator_meta = {
        'type_s': 'indicator',
        'start_date_dt': np.datetime_as_string(indicator_ds.time.values[0], unit='s'),
        'end_date_dt': np.datetime_as_string(indicator_ds.time.values[-1], unit='s'),
        'modified_time_dt': chk_time,
        'indicator_filename_s': indicator_filename,
        'indicator_filepath_s': str(indicator_output_path),
        'indicator_checksum_s': md5(indicator_output_path),
        'indicator_file_size_l': indicator_output_path.stat().st_size,
        'patterns_anoms_filename_s': patterns_anoms_filename,
        'patterns_anoms_filepath_s': str(patterns_and_anoms_output_path),
        'patterns_anoms_checksum_s': md5(patterns_and_anoms_output_path),
        'patterns_anoms_file_size_l': patterns_and_anoms_output_path.stat().st_size,
        'globals_filename_s': global_filename,
        'globals_filepath_s': str(global_output_path),
        'globals_checksum_s': md5(global_output_path),
        'globals_file_size_l': global_output_path.stat().st_size
    }

    if update:
        indicator_meta['id'] = indicator_query[0]['id']

    # Update Solr with dataset metadata
    resp = solr_update(config, [indicator_meta])

    if resp.status_code == 200:
        print('\nSuccessfully created or updated Solr index document')
    else:
        print('\nFailed to create or update Solr index document')
