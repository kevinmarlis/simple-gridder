"""

"""
import os
import sys
import hashlib
import logging
import warnings
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from numpy.lib.utils import source
import pyresample as pr
import requests
import xarray as xr
import xesmf as xe
from netCDF4 import default_fillvals
from pyresample.utils import check_and_wrap
from scipy.optimize import leastsq
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


def calc_grid_climate_index(agg_ds, pattern, pattern_ds, ann_cyc_in_pattern, ecco_latlon_grid, mapping_dir, method=3):
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

    generalized_functions_path = Path(
        f'{Path(__file__).resolve().parents[4]}/SLI-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    # TODO: this could be problematic if the first case is not a np.datetime64 object
    # not sure if second case is needed (could be from an early cycle draft)
    # extract center time of this agg field
    if 'cycle_center' in agg_ds.attrs:
        center_time = agg_ds.Time.values

    elif 'time_center' in agg_ds.attrs:
        center_time = np.datetime64(agg_ds.time_center)

    # determine its month
    agg_ds_center_mon = int(str(center_time)[5:7])

    pattern_field = pattern_ds[pattern][f'{pattern}_pattern'].values

    # Check if nearest_source_index has already been calculated for this pattern,
    # open if it has, otherwise run it and save it.

    pattern_mapping_dir = mapping_dir / pattern
    pattern_mapping_dir.mkdir(parents=True, exist_ok=True)
    source_indices_fp = pattern_mapping_dir / f'{pattern}_source_indices.p'
    num_sources_fp = pattern_mapping_dir / f'{pattern}_num_sources.p'

    agg_lons = agg_ds.Longitude.values
    agg_lats = agg_ds.Latitude.values

    pattern_lons = pattern_ds[pattern].Longitude.values
    pattern_lats = pattern_ds[pattern].Latitude.values

    if not os.path.exists(source_indices_fp) or not os.path.exists(num_sources_fp):
        temp_agg_lons, temp_agg_lats = check_and_wrap(agg_lons,
                                                      agg_lats)
        agg_lons_m, agg_lats_m = np.meshgrid(temp_agg_lons,
                                             temp_agg_lats)
        agg_swath_def = pr.geometry.SwathDefinition(lons=agg_lons_m,
                                                    lats=agg_lats_m)

        temp_pattern_lons, temp_pattern_lats = check_and_wrap(pattern_lons,
                                                              pattern_lats)
        pattern_lons_m, pattern_lats_m = np.meshgrid(temp_pattern_lons,
                                                     temp_pattern_lats)
        pattern_swath_def = pr.geometry.SwathDefinition(lons=pattern_lons_m,
                                                        lats=pattern_lats_m)

        # TODO: ask Ian about how to calculate target_grid_radius for patterns
        # instead of using ecco's grid area
        pattern_grid_radius = np.sqrt(
            ecco_latlon_grid.area.values.ravel())/2*np.sqrt(2)

        # find_mappings_from_source_to_target(source_grid, target_grid, target_grid_radius, source_grid_min_L, source_grid_max_L, neighbours=100, less_output=True)
        source_indices_within_target_radius_i, num_source_indices_within_target_radius_i, nearest_source_index_to_target_index_i = ea.find_mappings_from_source_to_target(agg_swath_def,
                                                                                                                                                                          pattern_swath_def,
                                                                                                                                                                          pattern_grid_radius,
                                                                                                                                                                          2e4,
                                                                                                                                                                          11e4,
                                                                                                                                                                          neighbours=1)
        with open(source_indices_fp, "wb") as f:
            pickle.dump(source_indices_within_target_radius_i, f)
        with open(num_sources_fp, "wb") as f:
            pickle.dump(num_source_indices_within_target_radius_i, f)
    else:
        with open(source_indices_fp, "rb") as f:
            source_indices_within_target_radius_i = pickle.load(f)
        with open(num_sources_fp, "rb") as f:
            num_source_indices_within_target_radius_i = pickle.load(f)

    pattern_vals = agg_ds[f'SSHA_{pattern}_removed_global_linear_trend'].values
    pattern_vals_1d = pattern_vals.ravel()
    # new_vals_1d = np.ndarray(len(ecco_latlon_grid.area.values.ravel()),)
    new_vals_1d = np.ndarray(len(pattern_field.ravel()),)

    print(f'\tMapping source to {pattern} grid.')
    for i in range(len(num_source_indices_within_target_radius_i)):

        if num_source_indices_within_target_radius_i[i] != 0:
            new_vals_1d[i] = sum(pattern_vals_1d[source_indices_within_target_radius_i[i]]
                                 ) / num_source_indices_within_target_radius_i[i]

    new_vals = new_vals_1d.reshape(
        len(pattern_lats), len(pattern_lons))

    ssha_to_pattern_da = xr.DataArray(new_vals, dims=['Latitude', 'Longitude'],
                                      coords={'Latitude': pattern_lats,
                                              'Longitude': pattern_lons, })

    ssha_to_pattern_da = agg_ds[f'SSHA_{pattern}_removed_global_linear_trend']

    ssha_to_pattern_da = ssha_to_pattern_da.assign_coords(
        coords={'time': center_time})

    # ds_in = agg_ds[f'SSHA_{pattern}_removed_global_linear_trend'].rename(
    #     {'Longitude': 'lon', 'Latitude': 'lat'}).drop(['longitude', 'latitude', 'Z', 'degree', 'Time'])

    # ds_out = pattern_ds[pattern][f'{pattern}_pattern'].rename(
    #     {'Longitude': 'lon', 'Latitude': 'lat'})

    # print(ds_in)
    # print(ds_out)

    # weights_dir = '/Users/kevinmarlis/Developer/JPL/sealevel_output/indicator/weights/'
    # weight_fp = f'{weights_dir}/1812_to_{pattern}.nc'
    # if not os.path.exists(weight_fp):
    #     print(f'Creating {pattern} weight file.')

    # # HiddenPrints class keeps xesmf.Regridder from printing details about weights
    # with HiddenPrints():
    #     regridder = xe.Regridder(
    #         ds_in, ds_out, 'bilinear', filename=weight_fp, reuse_weights=True)

    # ssha_to_pattern_da = regridder(ds_in)
    # print('here')

    # ssha_to_pattern_da = ssha_to_pattern_da.assign_coords(
    #     coords={'time': center_time})
    # ssha_to_pattern_da.to_netcdf('regridder_version.nc')
    # exit()

    # remove the monthly mean pattern from the gridded ssha
    # now ssha_anom is w.r.t. seasonal cycle and MDT
    ssha_anom = ssha_to_pattern_da.values - \
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

    lats = ssha_to_pattern_da.Latitude.values
    lons = ssha_to_pattern_da.Longitude.values

    ssha_anom = xr.DataArray(ssha_anom, dims=['latitude', 'longitude'],
                             coords={'longitude': lons,
                                     'latitude': lats})

    return LS_result, center_time, ssha_anom


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

    pattern_field = pattern_ds[pattern][f'{pattern}_pattern']

    ssha_anom = agg_ds[f'SSHA_{pattern}_removed_global_linear_trend'] - \
        ann_cyc_in_pattern[pattern].ann_pattern.sel(
            month=agg_ds_center_mon)/1e3

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
    ssha_to_fit = agg_ds[f'SSHA_{pattern}_removed_global_linear_trend'].values[nn]

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

        ssha_grid = pr.geometry.SwathDefinition(
            lons=tmp_ssha_lons, lats=tmp_ssha_lats)

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
    generalized_functions_path = Path(
        f'{Path(__file__).resolve().parents[4]}/SLI-utils/')
    sys.path.append(str(generalized_functions_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    output_dir = output_path / 'indicator'
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
    fq = ['type_s:cycle', 'processing_success_b:true',
          f'processing_time_dt:[{modified_time} TO NOW]']

    # TODO: this is for development
    # gridded cycles through 2016, along track from 2017 - 2019
    # fq = ['(type_s:cycle AND dataset_s:*rads* AND processing_success_b:true)']
    fq = ['(type_s:cycle AND dataset_s:*rads* AND processing_success_b:true) OR (type_s:cycle AND \
    dataset_s:*1812* AND start_date_dt:[2014-01-01T00:00:00Z TO 2017-01-01T00:00:00Z] AND processing_success_b:true)']

    fq = ['type_s:cycle AND dataset_s:*1812* AND start_date_dt:[2014-01-01T00:00:00Z TO 2017-01-01T00:00:00Z] AND processing_success_b:true']

    updated_cycles = solr_query(config, fq, sort='start_date_dt asc')

    # ONLY PROCEED IF THERE ARE CYCLES NEEDING CALCULATING
    if not updated_cycles:
        print('No cycles modified since last index calculation.')
        return

    # TODO: Modify to select gridded data when available
    updated_cycle_dict = defaultdict(list)
    for updated_cycle in updated_cycles:
        updated_cycle_dict[updated_cycle['start_date_dt']].append(updated_cycle)

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
    mapping_dir = output_dir / 'mappings'
    mapping_dir.mkdir(parents=True, exist_ok=True)

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
    lon_m, lat_m = np.meshgrid(ann_ds.Longitude.values, ann_ds.Latitude.values)
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

    for key in updated_cycle_dict.keys():
        cycles_in_range = updated_cycle_dict[key]

        if len(cycles_in_range) == 1:
            cycle = cycles_in_range[0]
            cycle_ds = xr.open_dataset(cycle['filepath_s'])
        else:
            all_at_cycle = True
            for cycle_meta in cycles_in_range:
                if '1812' in cycle_meta['dataset_s']:
                    cycle = cycle_meta
                    cycle_ds = xr.open_dataset(cycle['filepath_s'])
                    all_at_cycle = False
            if all_at_cycle:
                ats = []
                for cycle_meta in cycles_in_range:
                    ds = xr.open_dataset(cycle_meta['filepath_s'])
                    ats.append(ds)

                cycle_ds = xr.concat(ats, 'Time')
                cycle_ds = cycle_ds.sortby('Time')
                cycle = cycle_meta

        cycle_ds.close()
        cycle_type = cycle['index_type_s']
        date = cycle['filepath_s'][-18:-10]
        date = f'{date[:4]}-{date[4:6]}-{date[6:8]}'

        print(f' - Calculating index values for {date}')

        if 'cycle_center' in cycle_ds.attrs:
            ct = np.datetime64(cycle_ds.cycle_center)

        elif 'time_center' in cycle_ds.attrs:
            ct = np.datetime64(cycle_ds.time_center)

        if cycle_type == 'along_track':
            # Use roi=2e5, sigma=1e5, neighbours=1000 for final version
            ssha_lon_nn, ssha_lat_nn, sha_nn, ssha_to_pattern_da = interp_ssha_points_to_pattern(pattern_area_defs['global'],
                                                                                                 cycle_ds.SSHA.values.ravel(),
                                                                                                 cycle_ds.SSHA.Longitude.values.ravel(),
                                                                                                 cycle_ds.SSHA.Latitude.values.ravel(),
                                                                                                 roi=4e5, sigma=1e5, neighbours=5)
        # Gridded cycles
        else:
            global_mapping_dir = mapping_dir / 'global'
            global_mapping_dir.mkdir(parents=True, exist_ok=True)
            source_indices_fp = global_mapping_dir / 'half_deg_global_source_indices.p'
            num_sources_fp = global_mapping_dir / 'half_deg_global_num_sources.p'

            if not os.path.exists(source_indices_fp) or not os.path.exists(num_sources_fp):
                print('here')
                # Create index mappings from 1812 cycle to ECCO grid
                cycle_lons = cycle_ds.Longitude.values
                cycle_lats = cycle_ds.Latitude.values

                temp_pattern_lons, temp_pattern_lats = check_and_wrap(cycle_lons,
                                                                      cycle_lats)

                cycle_lons_m, cycle_lats_m = np.meshgrid(temp_pattern_lons,
                                                         temp_pattern_lats)

                cycle_swath_def = pr.geometry.SwathDefinition(lons=cycle_lons_m,
                                                              lats=cycle_lats_m)

                ecco_grid_swath_def = pattern_area_defs['global']

                ecco_grid_radius = np.sqrt(
                    ecco_latlon_grid.area.values.ravel())/2*np.sqrt(2)

                source_indices_within_target_radius_i, num_source_indices_within_target_radius_i, nearest_source_index_to_target_i = ea.find_mappings_from_source_to_target(cycle_swath_def,
                                                                                                                                                                            ecco_grid_swath_def,
                                                                                                                                                                            ecco_grid_radius,
                                                                                                                                                                            3e3,
                                                                                                                                                                            20e3)
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
                Time=cycle_ds.Time.values[0]).SSHA.values.T
            cycle_vals_1d = cycle_vals.ravel()
            new_vals_1d = np.ndarray(len(ecco_latlon_grid.area.values.ravel()),)

            print('\tMapping source to target grid.')
            for i in range(len(num_source_indices_within_target_radius_i)):
                if num_source_indices_within_target_radius_i[i] != 0:

                    new_vals_1d[i] = sum(cycle_vals_1d[source_indices_within_target_radius_i[i]]
                                         ) / num_source_indices_within_target_radius_i[i]
                else:
                    new_vals_1d[i] = np.nan

            # The regridded data
            new_vals = new_vals_1d.reshape(len(global_lat), len(global_lon))

        global_da = xr.DataArray(new_vals, dims=['latitude', 'longitude'],
                                 coords={'longitude': global_lon,
                                         'latitude': global_lat})

        global_da = global_da.assign_coords(
            coords={'Time': np.datetime64(cycle_ds.cycle_center)})

        global_dam = global_da.where(ecco_latlon_grid.maskC.isel(Z=0) > 0)
        global_dam.name = 'SSHA_GLOBAL'
        global_dam.attrs = {
            'comment': 'Global SSHA land masked using ECCO lat/lon grid'}
        global_dsm = global_dam.to_dataset()

        # Spatial Mean
        nzp = np.where(~np.isnan(global_dam), 1, np.nan)
        area_nzp = np.sum(nzp * ecco_latlon_grid.area)
        spatial_mean = float(
            np.nansum(global_dam * ecco_latlon_grid.area) / area_nzp)
        mean_da = xr.DataArray(spatial_mean, coords={'time': ct})
        mean_da.name = 'spatial_mean'
        mean_da.attrs = {'comment': 'Global SSHA spatial mean'}

        global_dam_removed_mean = global_dam - spatial_mean
        global_dam_removed_mean.attrs = {
            'comment': 'Global SSHA with global spatial mean removed'}
        global_dsm['SSHA_GLOBAL_removed_global_spatial_mean'] = global_dam_removed_mean

        # Linear Trend
        trend_ds = xr.open_dataset(bh_dir / 'pointwise_sealevel_trend.nc')

        time_diff = (global_da.Time.values - np.datetime64('1992-10-02')
                     ).astype(np.int32)/1e9

        trend = (time_diff * trend_ds['pointwise_sealevel_trend']) + \
            trend_ds['pointwise_sealevel_offset']

        global_dam_detrended = global_dam - trend
        global_dam_detrended.attrs = {
            'comment': 'Global SSHA with linear trend removed'}
        global_dsm['SSHA_GLOBAL_removed_linear_trend'] = global_dam_detrended

        global_dsm = global_dsm.drop_vars('Z')
        global_dsm = global_dsm.drop(['degree'])

        method = 3

        indicators_agg_das = {}
        offset_agg_das = {}
        pattern_and_anom_das = {}

        for pattern in patterns:
            pattern_lats = pattern_ds[pattern]['Latitude']
            pattern_lons = pattern_ds[pattern]['Longitude']
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

            agg_da = global_dam_detrended.sel(
                longitude=pattern_lons, latitude=pattern_lats)
            agg_da.name = f'SSHA_{pattern}_removed_global_linear_trend'
            agg_da = agg_da.assign_coords(
                coords={'Time': np.datetime64(cycle_ds.cycle_center)})
            try:
                agg_da = agg_da.drop(['degree'])
            except:
                pass

            agg_ds = agg_da.to_dataset()
            agg_ds.attrs = cycle_ds.attrs

            if cycle_type == 'along_track':
                index_calc, ct,  ssha_anom = calc_at_climate_index(agg_ds, pattern, pattern_ds,
                                                                   ann_cyc_in_pattern,
                                                                   method=method)
            else:
                index_calc, ct, ssha_anom = calc_grid_climate_index(agg_ds, pattern, pattern_ds,
                                                                    ann_cyc_in_pattern, ecco_latlon_grid,
                                                                    mapping_dir, method=method)
                ssha_anom = ssha_anom.rename(
                    {'latitude': 'Latitude', 'longitude': 'Longitude'})

            anom_name = f'SSHA_{pattern}_removed_global_linear_trend_and_seasonal_cycle'
            ssha_anom.name = anom_name
            ssha_anom.attrs = {}
            agg_ds[anom_name] = ssha_anom

            try:
                agg_ds = agg_ds.drop_vars(['Z', 'latitude', 'longitude'])
                agg_ds = agg_ds.drop_dims(['latitude', 'longitude'])
            except Exception:
                pass
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
        all_patterns_and_anoms = []

        for pattern in patterns:
            indicators_agg_da = indicators_agg_das[pattern]
            offsets_agg_da = offset_agg_das[pattern]

            all_indicators.append(
                xr.merge([offsets_agg_da, indicators_agg_da, mean_da]))

            all_patterns_and_anoms.append(pattern_and_anom_das[pattern])

        # FINISHED THROUGH ALL PATTERNS
        # append all the datasets together into a single dataset to rule them all
        indicator_ds = xr.merge(all_indicators)
        indicator_ds = indicator_ds.expand_dims(Time=[indicator_ds.time.values])
        indicator_ds = indicator_ds.drop(['time'])
        pattern_anoms_ds = xr.merge(all_patterns_and_anoms)
        globals_ds = global_dsm

        fp_date = date.replace('-', '_')
        # date =

        indicator_filename = f'{fp_date}_indicator.nc'
        pattern_anoms_filename = f'{fp_date}_pattern_ssha_anoms.nc'
        global_filename = f'{fp_date}_globals.nc'

        cycle_indicators_path = output_dir / 'cycle_indicators'
        cycle_indicators_path.mkdir(parents=True, exist_ok=True)
        indicator_output_path = cycle_indicators_path / indicator_filename

        cycle_pattern_anoms_path = output_dir / 'cycle_pattern_anoms'
        cycle_pattern_anoms_path.mkdir(parents=True, exist_ok=True)
        pattern_anoms_output_path = cycle_pattern_anoms_path / pattern_anoms_filename

        cycle_globals_path = output_dir / 'cycle_globals'
        cycle_globals_path.mkdir(parents=True, exist_ok=True)
        global_output_path = cycle_globals_path / global_filename

        all_ds = [(indicator_ds, indicator_output_path),
                  (pattern_anoms_ds, pattern_anoms_output_path),
                  (globals_ds, global_output_path)]

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

    cycle_indicators_path = output_dir / 'cycle_indicators'
    indicator_ds = xr.open_mfdataset(f'{cycle_indicators_path}/*.nc')
    indicator_ds.to_netcdf(output_dir / 'complete_indicator_ds.nc')
    print(indicator_ds)
    exit()

    # # Concatenate the list of individual DAs along time
    # # Merge into a single DataSet and append that pattern to all_indicators list
    # # List to hold DataSet objects for each pattern
    # all_indicators = []
    # all_patterns_and_anoms = []

    # print('Combining patterns...')
    # for pattern in patterns:
    #     print('...')
    #     indicators_agg_da = indicators_agg_das[pattern]
    #     offsets_agg_da = offset_agg_das[pattern]

    #     all_indicators.append(
    #         xr.merge([offsets_agg_da, indicators_agg_da, spatial_mean_da]))

    #     all_patterns_and_anoms.append(pattern_and_anom_das[pattern])

    # # FINISHED THROUGH ALL PATTERNS
    # # append all the datasets together into a single dataset to rule them all
    # print('Merging indicators')
    # new_indicators = xr.merge(all_indicators)
    # print('Merging patterns and anoms')
    # new_patterns_and_anoms = xr.merge(all_patterns_and_anoms)
    # print('Merging globals')
    # new_globals = global_ds

    # # Open existing indicator ds to add new values if needed
    # if update:
    #     print('\nAdding calculated values to indicator netCDF.')

    #     # PROCESS INDICATORS
    #     indicator_ds = xr.open_dataset(
    #         indicator_metadata['indicator_filepath_s'])

    #     # Remove times of new indicator values from original indicator DS if they exist
    #     # (this effectively updates the values)
    #     indicator_ds = indicator_ds.where(~indicator_ds['time'].isin(
    #         np.unique(new_indicators['time'])), drop=True)

    #     # And use xr.concat to add in the new values (concat will create multiple entries for the
    #     # same time value).
    #     indicator_ds = xr.concat([indicator_ds, new_indicators], 'time')

    #     # Finally, sort to get things in the right order
    #     indicator_ds = indicator_ds.sortby('time')

    #     # Reapeat for other files

    #     # PROCESS PATTERNS AND ANOMS
    #     patterns_and_anoms_ds = xr.open_dataset(
    #         indicator_metadata['patterns_anoms_filepath_s'])
    #     # Load is required for sorting
    #     patterns_and_anoms_ds.load()
    #     patterns_and_anoms_ds = patterns_and_anoms_ds.where(~patterns_and_anoms_ds['Time'].isin(
    #         np.unique(new_patterns_and_anoms['Time'])), drop=True)
    #     patterns_and_anoms_ds = xr.concat(
    #         [patterns_and_anoms_ds, new_patterns_and_anoms], 'Time')
    #     patterns_and_anoms_ds = patterns_and_anoms_ds.sortby('Time')

    #     # PROCESS GLOBALS
    #     globals_ds = xr.open_dataset(indicator_metadata['globals_filepath_s'])
    #     # Load is required for sorting
    #     globals_ds.load()
    #     globals_ds = globals_ds.where(~globals_ds['Time'].isin(
    #         np.unique(new_globals['Time'])), drop=True)
    #     globals_ds = xr.concat([globals_ds, new_globals], 'Time')
    #     globals_ds = globals_ds.sortby('Time')

    # else:
    #     indicator_ds = new_indicators
    #     patterns_and_anoms_ds = new_patterns_and_anoms
    #     globals_ds = new_globals

    # # # NetCDF encoding
    # # encoding_each = {'zlib': True,
    # #                  'complevel': 5,
    # #                  'dtype': 'float32',
    # #                  'shuffle': True,
    # #                  '_FillValue': default_fillvals['f8']}

    # # coord_encoding = {}
    # # for coord in indicator_ds.coords:
    # #     coord_encoding[coord] = {'_FillValue': None,
    # #                              'dtype': 'float32',
    # #                              'complevel': 6}

    # #     if 'Time' in coord:
    # #         coord_encoding[coord] = {'_FillValue': None,
    # #                                  'zlib': True,
    # #                                  'contiguous': False,
    # #                                  'shuffle': False}

    # # var_encoding = {var: encoding_each for var in indicator_ds.data_vars}

    # # encoding = {**coord_encoding, **var_encoding}

    # indicator_ds.to_netcdf(indicator_output_path)
    # patterns_and_anoms_ds.to_netcdf(patterns_and_anoms_output_path)
    # globals_ds.to_netcdf(global_output_path)

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

    # # Update Solr with dataset metadata
    # resp = solr_update(config, [indicator_meta])

    # if resp.status_code == 200:
    #     print('\nSuccessfully created or updated Solr index document')
    # else:
    #     print('\nFailed to create or update Solr index document')
