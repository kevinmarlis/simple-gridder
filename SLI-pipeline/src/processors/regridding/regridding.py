"""
"""

import hashlib
import logging
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
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module
from pyresample.kd_tree import resample_gauss
from pyresample.utils import check_and_wrap

log = logging.getLogger(__name__)
log.setLevel(logging.ERROR)

warnings.filterwarnings("ignore")


def md5(fpath):
    """
    Creates md5 checksum from file

    Params:
        fpath (str): path of the file

    Returns:
        hash_md5.hexdigest (str): double length string containing only hexadecimal digits
    """
    hash_md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def solr_query(config, fq):
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
                    'sort': 'date_dt asc'}

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


def regridder(cycle_ds, data_type, output_dir, method='gaussian', ats=[], neighbours=500):
    ea_path = Path(f'{Path(__file__).resolve().parents[4]}/SLI-utils/')
    sys.path.append(str(ea_path))
    import ecco_cloud_utils as ea  # pylint: disable=import-error

    date_regex = '%Y-%m-%dT%H:%M:%S'

    mapping_dir = output_dir / 'mappings'
    mapping_dir.mkdir(parents=True, exist_ok=True)

    ref_files_path = Path().resolve() / 'SLI-pipeline' / 'ref_files'

    global_path = ref_files_path / 'GRID_GEOMETRY_ECCO_V4r4_latlon_0p50deg.nc'
    global_ds = xr.open_dataset(global_path)

    wet_ins = np.where(global_ds.maskC.isel(Z=0).values.ravel() > 0)[0]

    global_mapping_dir = mapping_dir / 'global'
    global_mapping_dir.mkdir(parents=True, exist_ok=True)

    global_lon = global_ds.longitude.values
    global_lat = global_ds.latitude.values

    global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)
    target_lons_wet = global_lon_m.ravel()[wet_ins]
    target_lats_wet = global_lat_m.ravel()[wet_ins]

    global_swath_def = pr.geometry.SwathDefinition(lons=target_lons_wet,
                                                   lats=target_lats_wet)

    if data_type == 'gridded':

        global_attrs = cycle_ds.attrs
        var_attrs = cycle_ds['SSHA'].attrs

        source_indices_fp = global_mapping_dir / 'half_deg_global_source_indices.p'
        num_sources_fp = global_mapping_dir / 'half_deg_global_num_sources.p'

        if not source_indices_fp.exists() or not num_sources_fp.exists():
            # Create index mappings from 1812 cycle to ECCO grid
            cycle_lons = cycle_ds.longitude.values
            cycle_lats = cycle_ds.latitude.values

            temp_pattern_lons, temp_pattern_lats = check_and_wrap(cycle_lons,
                                                                  cycle_lats)

            cycle_lons_m, cycle_lats_m = np.meshgrid(temp_pattern_lons,
                                                     temp_pattern_lats)

            cycle_swath_def = pr.geometry.SwathDefinition(lons=cycle_lons_m,
                                                          lats=cycle_lats_m)

            global_grid_radius = np.sqrt(
                global_ds.area.values.ravel())/2*np.sqrt(2)

            global_grid_radius_wet = global_grid_radius.ravel()[wet_ins]

            # Used for development
            # neighbours = 1

            source_indices_within_target_radius_i, num_source_indices_within_target_radius_i, nearest_source_index_to_target_i = ea.find_mappings_from_source_to_target(cycle_swath_def,
                                                                                                                                                                        global_swath_def,
                                                                                                                                                                        global_grid_radius_wet,
                                                                                                                                                                        100,
                                                                                                                                                                        20e3,
                                                                                                                                                                        neighbours=neighbours)
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
            time=cycle_ds.time.values[0]).SSHA.values.T
        cycle_vals_1d = cycle_vals.ravel()

        new_vals = np.zeros_like(global_ds.area.values) * np.NaN

        print('\tMapping source to target grid.')
        for i in range(len(num_source_indices_within_target_radius_i)):
            if num_source_indices_within_target_radius_i[i] != 0:

                new_vals.ravel()[wet_ins[i]] = sum(cycle_vals_1d[source_indices_within_target_radius_i[i]]
                                                   ) / num_source_indices_within_target_radius_i[i]

        valid_keys = [k for k in cycle_ds['latitude'].attrs.keys()
                      if 'valid' in k]
        for k in valid_keys:
            del cycle_ds['latitude'].attrs[k]
            del cycle_ds['longitude'].attrs[k]

    # Along track data
    else:

        instr_in_cycle = [ds.attrs['original_dataset_short_name']
                          for ds in ats]
        global_attrs = cycle_ds.attrs
        for key in [attr for attr in global_attrs.keys() if 'original' in attr]:
            global_attrs.pop(key)

        global_attrs['source'] = 'Combination of ' + \
            ', '.join(instr_in_cycle) + ' along track instruments'

        if method == 'gaussian':
            # Define the 'swath' as the lats/lon pairs of the model grid
            ssha_lon = cycle_ds.longitude.values.ravel()
            ssha_lat = cycle_ds.latitude.values.ravel()
            ssha = cycle_ds.SSHA.values.ravel()

            ssha_lat_nn = ssha_lat[~np.isnan(ssha)]
            ssha_lon_nn = ssha_lon[~np.isnan(ssha)]
            ssha_nn = ssha[~np.isnan(ssha)]

            if np.sum(~np.isnan(ssha_nn)) > 0:
                tmp_ssha_lons, tmp_ssha_lats = check_and_wrap(ssha_lon_nn.ravel(),
                                                              ssha_lat_nn.ravel())

                ssha_grid = pr.geometry.SwathDefinition(
                    lons=tmp_ssha_lons, lats=tmp_ssha_lats)

                roi = 6e5  # 6e5
                sigma = 1e5
                neighbours = 500  # 500

                new_vals = resample_gauss(ssha_grid, ssha_nn,
                                          global_swath_def,
                                          radius_of_influence=roi,
                                          sigmas=sigma,
                                          fill_value=np.NaN, neighbours=neighbours)

                new_vals_2d = np.zeros_like(global_ds.area.values) * np.nan
                for i, val in enumerate(new_vals):
                    new_vals_2d.ravel()[wet_ins[i]] = val
                new_vals = new_vals_2d

        else:
            cycle_lons = cycle_ds.longitude.values
            cycle_lats = cycle_ds.latitude.values

            temp_pattern_lons, temp_pattern_lats = check_and_wrap(cycle_lons,
                                                                  cycle_lats)

            cycle_lons_m, cycle_lats_m = np.meshgrid(temp_pattern_lons,
                                                     temp_pattern_lats)

            cycle_swath_def = pr.geometry.SwathDefinition(lons=cycle_lons_m,
                                                          lats=cycle_lats_m)

            global_lon_m, global_lat_m = np.meshgrid(global_lon, global_lat)
            target_lons_wet = global_lon_m.ravel()[wet_ins]
            target_lats_wet = global_lat_m.ravel()[wet_ins]

            global_swath_def = pr.geometry.SwathDefinition(lons=target_lons_wet,
                                                           lats=target_lats_wet)

            global_grid_radius = np.sqrt(
                global_ds.area.values.ravel())/2*np.sqrt(2)

            global_grid_radius_wet = global_grid_radius.ravel()[wet_ins]

            # Used for development
            # neighbours = 1

            source_indices_within_target_radius_i, num_source_indices_within_target_radius_i, nearest_source_index_to_target_i = ea.find_mappings_from_source_to_target(cycle_swath_def,
                                                                                                                                                                        global_swath_def,
                                                                                                                                                                        global_grid_radius_wet,
                                                                                                                                                                        100,
                                                                                                                                                                        20e3,
                                                                                                                                                                        neighbours=neighbours)

            cycle_vals = cycle_ds.SSHA.values.T
            cycle_vals_1d = cycle_vals.ravel()

            new_vals = np.zeros_like(global_ds.area.values) * np.nan

            print('\tMapping source to target grid.')
            for i in range(len(num_source_indices_within_target_radius_i)):
                if num_source_indices_within_target_radius_i[i] != 0:

                    new_vals.ravel()[wet_ins[i]] = sum(cycle_vals_1d[source_indices_within_target_radius_i[i]]
                                                       ) / num_source_indices_within_target_radius_i[i]

    regridded_da = xr.DataArray(new_vals, dims=['latitude', 'longitude'],
                                coords={'longitude': global_lon,
                                        'latitude': global_lat})

    regridded_da = regridded_da.assign_coords(
        coords={'time': np.datetime64(cycle_ds.cycle_center)})

    regridded_da.name = 'SSHA'
    regridded_ds = regridded_da.to_dataset()

    regridded_ds['mask'] = (['latitude', 'longitude'], np.where(
        global_ds['maskC'].isel(Z=0) == True, 1, 0))

    regridded_ds['mask'].attrs = {'long_name': 'wet/dry boolean mask for grid cell',
                                  'comment': '1 for ocean, otherwise 0'}

    regridded_ds.attrs = cycle_ds.attrs

    regridded_ds['SSHA'].attrs = cycle_ds['SSHA'].attrs
    regridded_ds['SSHA'].attrs['valid_min'] = np.nanmin(
        regridded_ds['SSHA'].values)
    regridded_ds['SSHA'].attrs['valid_max'] = np.nanmax(
        regridded_ds['SSHA'].values)
    regridded_ds['SSHA'].attrs['summary'] = 'Data gridded to 0.5 degree lat lon grid'

    regridded_ds['latitude'].attrs = cycle_ds['latitude'].attrs
    regridded_ds['longitude'].attrs = cycle_ds['longitude'].attrs

    if data_type == 'along_track' and method == 'gaussian':
        regridded_ds.attrs['gridding_method'] = \
            f'Gridded using pyresample resample_gauss with roi={roi}, neighbours={neighbours}'
    else:
        regridded_ds.attrs['gridding_method'] = \
            f'Gridded using find_mappings_from_source_to_target with neighbours={neighbours}'

    regridded_ds.attrs['comment'] = 'Gridded using ECCO V4r4 0.5 degree lat lon grid'

    return regridded_ds


def regridding(config, output_dir, reprocess, log_time):
    """
    Indicator pipeline uses three data sources: MEASuRES 1812, Jason 3, and Sentinel 3B.
    This function combines the two along track datasets on a single 0.5 degree latlon
    grid. It also regrids the gridded MEASuRES data to the same 0.5 degree latlon grid. 
    """
    # Set file handler for log using output_path
    formatter = logging.Formatter('%(asctime)s: %(message)s')

    logs_path = Path(output_dir / f'logs/{log_time}')
    logs_path.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(logs_path / 'regridder.log')
    file_handler.setLevel(logging.ERROR)
    file_handler.setFormatter(formatter)

    log.addHandler(file_handler)

    date_regex = '%Y-%m-%dT%H:%M:%S'
    version = config['version']

    regridding_status = True

    # ======================================================
    # Regrid measures cycles
    # ======================================================
    print('\nRegridding MEaSUREs cycles\n')

    existing_regridded_measures_cycles = defaultdict(list)
    for cycle in solr_query(config, ['type_s:regridded_cycle', 'processing_success_b:true', 'original_data_type_s:gridded']):
        start_date_key = cycle['start_date_dt'][:10]
        existing_regridded_measures_cycles[start_date_key].append(cycle)

    solr_gridded_cycles = solr_query(
        config, ['type_s:cycle', 'processing_success_b:true', 'data_type_s:gridded'])

    gridded_cycles = defaultdict(list)
    for cycle in solr_gridded_cycles:
        date_key = cycle['start_date_dt'][:10]
        gridded_cycles[date_key].append(cycle)

    for date, cycle in gridded_cycles.items():
        cycle = cycle[0]
        # only regrid if cycle was modified
        update = date not in existing_regridded_measures_cycles.keys() or \
            cycle['processing_time_dt'] > existing_regridded_measures_cycles[date][0]['processing_time_dt']

        if update:
            try:
                print(f'Regridding MEaSUREs cycle {date}')
                cycle_ds = xr.open_dataset(cycle['filepath_s'])
                regridded_ds = regridder(cycle_ds, 'gridded', output_dir)

                regrid_dir = output_dir / 'regridded_cycles/MEASURES_1812'
                regrid_dir.mkdir(parents=True, exist_ok=True)

                center_date = cycle["center_date_dt"]

                filename = f'ssha_global_half_deg_{center_date[:10].replace("-", "_")}.nc'
                global_fp = regrid_dir / filename
                encoding = cycle_ds_encoding(regridded_ds)

                regridded_ds.to_netcdf(global_fp, encoding=encoding)

                # Determine checksum and file size
                checksum = md5(global_fp)
                file_size = global_fp.stat().st_size
                processing_success = True
            except:
                log.exception(f'\nError while processing cycle {date}')
                filename = ''
                global_fp = ''
                checksum = ''
                file_size = 0
                processing_success = False
                regridding_status = False

            item = cycle
            item.pop('id')
            item.pop('dataset_s')
            item.pop('_version_')
            item['type_s'] = 'regridded_cycle'
            item['combination_s'] = 'MEASURES_1812'
            item['filename_s'] = filename
            item['filepath_s'] = str(global_fp)
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['processing_success_b'] = processing_success
            item['processing_time_dt'] = datetime.utcnow().strftime(date_regex)
            item['processing_version_f'] = version
            item['checksum_s'] = checksum
            item['original_data_type_s'] = item.pop('data_type_s')

            if date in existing_regridded_measures_cycles.keys():
                item['id'] = existing_regridded_measures_cycles[date][0]['id']

            resp = solr_update(config, [item])
            if resp.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')
            else:
                print('\tFailed to create Solr cycle documents')

        else:
            print(f'\tNo updates to regridded MEaSUREs cycle {date}')

    # ======================================================
    # Regrid along track cycles
    # ======================================================
    print('\nRegridding along track cycles\n')

    existing_regridded_at_cycles = {}
    for cycle in solr_query(config, ['type_s:regridded_cycle', 'combination_s:DAILY']):
        start_date_key = cycle['start_date_dt'][:10]
        existing_regridded_at_cycles[start_date_key] = cycle

    # Start with J3 cycles
    fq = ['type_s:cycle', 'processing_success_b:true', 'dataset_s:JASON_3_again']
    j3_cycles = solr_query(config, fq)

    for j3_cycle in j3_cycles:
        j3_ds = xr.open_dataset(j3_cycle['filepath_s'])

        at_cycles = [j3_ds]

        j3_start = j3_cycle['start_date_dt']
        j3_end = j3_cycle['end_date_dt']

        # Find S3 cycle with time overlap
        # J3 start_date >= S3 start_date and J3 start_date <= S3 end_date
        # S3 start_date >= J3 start_date and J3 end_date <= S3 end_date
        fq = ['type_s:cycle', 'processing_success_b:true', 'dataset_s:SENTINEL_3B',
              f'start_date_dt:["1992-01-01T00:00:00Z" TO {j3_start}]', f'end_date_dt:{{{j3_start} TO NOW]']
        s3_cycles = solr_query(config, fq)

        update = False

        if j3_start[:10] in existing_regridded_at_cycles.keys():
            existing_regrid_meta = existing_regridded_at_cycles[j3_start[:10]]

            # Determine if J3 or S3 cycles have been updated
            if j3_cycle['processing_time_dt'] > existing_regrid_meta['processing_time_dt']:
                update = True

            if s3_cycles:
                for s3 in s3_cycles:
                    if s3['processing_time_dt'] > existing_regrid_meta['processing_time_dt']:
                        update = True

            if not existing_regrid_meta['processing_success_b']:
                update = True
        else:
            update = True

        if not update:
            print(f'\tNo updates to regridded DAILY cycle {j3_start}')
        else:

            print(f'Regridding DAILY cycle {j3_start}')
            try:
                s3_data = []
                if s3_cycles:
                    for s3 in s3_cycles:
                        s3_ds = xr.open_dataset(s3['filepath_s'])

                        try:
                            s3_slice_ds = s3_ds.sel(
                                time=slice(j3_start, j3_end))
                        except:
                            temp_start = j3_start[:-1]
                            temp_end = j3_end[:-1]

                            s3_slice_ds = s3_ds.sel(
                                time=slice(temp_start, temp_end))

                        s3_data.append(s3_slice_ds)

                    #   Only need one S3 metadata added to list
                    at_cycles.append(s3_ds)

                all_data = [j3_ds]

                if s3_data:
                    s3_merged = xr.merge(s3_data)
                    all_data.append(s3_merged)

                all_data_ds = xr.concat(all_data, 'time')
                all_data_ds = all_data_ds.sortby('time')

                regridded_ds = regridder(
                    all_data_ds, 'along_track', output_dir, ats=at_cycles)

                regrid_dir = output_dir / 'regridded_cycles/DAILY_test'
                regrid_dir.mkdir(parents=True, exist_ok=True)

                filename = f'ssha_global_half_deg_{j3_start[:10].replace("-", "")}.nc'
                global_fp = regrid_dir / filename
                encoding = cycle_ds_encoding(regridded_ds)

                regridded_ds.to_netcdf(global_fp, encoding=encoding)

                # Determine checksum and file size
                checksum = md5(global_fp)
                file_size = global_fp.stat().st_size
                processing_success = True
            except Exception as e:
                log.exception(
                    f'\nError while processing cycle {date}. {e}')
                filename = ''
                global_fp = ''
                checksum = ''
                file_size = 0
                processing_success = False
                regridding_status = False

            item = {}
            item['type_s'] = 'regridded_cycle'
            item['combination_s'] = 'DAILY'
            item['start_date_dt'] = j3_cycle['start_date_dt']
            item['center_date_dt'] = j3_cycle['center_date_dt']
            item['end_date_dt'] = j3_cycle['end_date_dt']
            item['cycle_length_i'] = j3_cycle['cycle_length_i']
            item['filename_s'] = filename
            item['filepath_s'] = str(global_fp)
            item['checksum_s'] = checksum
            item['file_size_l'] = file_size
            item['processing_success_b'] = processing_success
            item['processing_time_dt'] = datetime.utcnow().strftime(
                date_regex)
            item['processing_version_f'] = version
            item['original_data_type_s'] = j3_cycle['data_type_s']

            if date in existing_regridded_at_cycles.keys():
                item['id'] = existing_regridded_at_cycles[date][0]['id']
            resp = solr_update(config, [item])
            if resp.status_code == 200:
                print(
                    '\tSuccessfully created or updated Solr cycle documents')
            else:
                print('\tFailed to create Solr cycle documents')

    if not regridding_status:
        raise Exception('One or more regriddings failed. Check Solr and logs.')

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Iterate through regridding combinations in config YAML
    # for combination in config['combinations']:
    #     combo_name = combination['name']
    #     print(combo_name)
    #     existing_regridded_at_cycles = defaultdict(list)
    #     for cycle in solr_query(config, ['type_s:regridded_cycle',
    #                                      'original_data_type_s:along_track',
    #                                      f'combination_s:{combo_name}']):
    #         start_date_key = cycle['start_date_dt'][:10]
    #         existing_regridded_at_cycles[start_date_key].append(cycle)

    #     if 'times' in combination.keys():
    #         inst_start = combination['times'][0]
    #         inst_end = combination['times'][1]

    #         instruments = combination['instruments']
    #     else:
    #         inst_start = combination[inst][0]
    #         inst_end = combination[inst][1]

    #         instruments = [k for k in combination.keys() if k != 'name']

    #     if inst_start == 'EARLIEST':
    #         inst_start = '1970-01-01'

    #     inst_start += 'T00:00:00Z'

    #     if inst_end == 'NOW':
    #         inst_end = datetime.utcnow().strftime(date_regex)
    #         inst_end += 'Z'
    #     else:
    #         inst_end += 'T00:00:00Z'

    #     cycles_to_regrid = []

    #     # Collect cycles within each instrument's date range
    #     for inst in instruments:
    #         inst_start = combination[inst][0]

    #         if inst_start == 'EARLIEST':
    #             inst_start = '1970-01-01'

    #         inst_start += 'T00:00:00Z'

    #         inst_end = combination[inst][1]

    #         if inst_end == 'NOW':
    #             inst_end = datetime.utcnow().strftime(date_regex)
    #             inst_end += 'Z'
    #         else:
    #             inst_end += 'T00:00:00Z'

    #         fq = ['type_s:cycle', 'processing_success_b:true', f'dataset_s:{inst}',
    #               f'start_date_dt:[{inst_start} TO {inst_end}]']
    #         cycles_in_range = solr_query(config, fq)

    #         cycles_to_regrid.extend(cycles_in_range)

    #     # Group together cycles across instruments by date
    #     along_track_cycles = defaultdict(list)
    #     for cycle in cycles_to_regrid:
    #         date_key = cycle['start_date_dt'][:10]
    #         along_track_cycles[date_key].append(cycle)

    #     dates = list(along_track_cycles.keys())
    #     dates.sort()

    #     # TODO: implement multiprocessing?

    #     # Iterate through dates and regrid all cycles that fall on that date
    #     for date in dates:
    #         cycles = along_track_cycles[date]
    #         update = False

    #         for cycle in cycles:
    #             fq = ['type_s:regridded_cycle',
    #                   f'combination_s:{combo_name}', f'start_date_dt:"{cycle["start_date_dt"]}"']
    #             solr_combo = solr_query(config, fq)

    #             if len(solr_combo) == 1:
    #                 existing_regrid_meta = solr_combo[0]

    #                 # if cycle was reprocessed after prior regridding
    #                 # or
    #                 # prior regridding failed
    #                 if cycle['processing_time_dt'] > existing_regrid_meta['processing_time_dt'] or not existing_regrid_meta['processing_success_b']:
    #                     update = True
    #                     break
    #             else:
    #                 update = True
    #                 break

    #         if update:
    #             try:
    #                 print(
    #                     f'Regridding {combo_name} along track cycle {date}')

    #                 ats = []

    #                 if len(cycles) == 1:
    #                     cycle = cycles[0]
    #                     cycle_ds = xr.open_dataset(cycle['filepath_s'])
    #                     ats.append(cycle_ds)

    #                 else:
    #                     for cycle_meta in cycles:
    #                         inst = cycle_meta['dataset_s']

    #                         start = combination[inst][0]
    #                         if start == 'EARLIEST':
    #                             start = '1970-01-01'
    #                         start += 'T00:00:00'
    #                         start = np.datetime64(start)

    #                         end = combination[inst][1]
    #                         if end == 'NOW':
    #                             end = datetime.utcnow().strftime(date_regex)
    #                         else:
    #                             end += 'T00:00:00'
    #                         end = np.datetime64(end)

    #                         ds = xr.open_dataset(cycle_meta['filepath_s'])

    #                         sel_dates = ds.time.values[(
    #                             ds.time >= start) & (ds.time < end)]

    #                         ds = ds.sel(time=sel_dates)

    #                         ats.append(ds)

    #                     cycle_ds = xr.concat(ats, 'time')
    #                     cycle_ds = cycle_ds.sortby('time')

    #                     cycle = cycle_meta

    #                 regridded_ds = regridder(
    #                     cycle_ds, 'along_track', output_dir, ats=ats)

    #                 regrid_dir = output_dir / f'regridded_cycles/{combo_name}'
    #                 regrid_dir.mkdir(parents=True, exist_ok=True)

    #                 start_date = cycle["start_date_dt"]
    #                 filename = f'ssha_global_half_deg_{start_date[:10].replace("-", "")}.nc'
    #                 global_fp = regrid_dir / filename
    #                 encoding = cycle_ds_encoding(regridded_ds)

    #                 regridded_ds.to_netcdf(global_fp, encoding=encoding)

    #                 # Determine checksum and file size
    #                 checksum = md5(global_fp)
    #                 file_size = global_fp.stat().st_size
    #                 processing_success = True
    #             except Exception as e:
    #                 log.exception(
    #                     f'\nError while processing cycle {date}. {e}')
    #                 filename = ''
    #                 global_fp = ''
    #                 checksum = ''
    #                 file_size = 0
    #                 processing_success = False
    #                 regridding_status = False

    #             item = {}
    #             item['type_s'] = 'regridded_cycle'
    #             item['combination_s'] = combo_name
    #             item['start_date_dt'] = cycle['start_date_dt']
    #             item['center_date_dt'] = cycle['center_date_dt']
    #             item['end_date_dt'] = cycle['end_date_dt']
    #             item['cycle_length_i'] = cycle['cycle_length_i']
    #             item['filename_s'] = filename
    #             item['filepath_s'] = str(global_fp)
    #             item['checksum_s'] = checksum
    #             item['file_size_l'] = file_size
    #             item['processing_success_b'] = processing_success
    #             item['processing_time_dt'] = datetime.utcnow().strftime(
    #                 date_regex)
    #             item['processing_version_f'] = version
    #             item['original_data_type_s'] = cycle['data_type_s']

    #             if date in existing_regridded_at_cycles.keys():
    #                 item['id'] = existing_regridded_at_cycles[date][0]['id']
    #             resp = solr_update(config, [item])
    #             if resp.status_code == 200:
    #                 print(
    #                     '\tSuccessfully created or updated Solr cycle documents')
    #             else:
    #                 print('\tFailed to create Solr cycle documents')

    #         else:
    #             print(
    #                 f'No updates to {combo_name} regridded along track cycle for {date}')

    # if not regridding_status:
    #     raise Exception('One or more regriddings failed. Check Solr and logs.')
