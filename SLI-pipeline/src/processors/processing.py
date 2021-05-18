"""
This module handles dataset processing into 10 day cycles.
"""

import logging
import hashlib
from datetime import datetime, timedelta
import requests
import numpy as np
import xarray as xr
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module

log = logging.getLogger(__name__)


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


def process_along_track(cycle_granules, ds_meta, dates):
    """
    Processes and aggregates individual granules that fall within a cycle's date range for
    a non GPS along track dataset.

    Params:
        cycle_granules (List[dict]): the dataset specific config file
        ds_meta (dict): the list of docs to update on Solr
        dates (Tuple[str, str, str]):

    Returns:
        cycle_ds (Dataset): the processed cycle Dataset object
        len(granules) (int): the number of granules within the processed cycle Dataset object
    """
    var = 'ssh_smoothed'
    reference_date = datetime(1985, 1, 1, 0, 0, 0)
    granules = []
    data_start_time = None
    data_end_time = None

    for granule in cycle_granules:
        ds = xr.open_dataset(granule['granule_file_path_s'], group='data')

        if 'gmss' in ds.data_vars:
            ds = ds.drop(['gmss'])
            ds = ds.rename_dims({'phony_dim_2': 'Time'})

        else:
            ds = ds.rename_dims({'phony_dim_1': 'Time'})

        ds = ds.rename_vars({'time': 'Time', var: 'SSHA'})
        ds = ds.rename({'lats': 'Latitude', 'lons': 'Longitude'})

        ds = ds.drop([var for var in ds.data_vars if var[0] == '_'])
        ds = ds.drop_vars(['ssh', 'sat_id', 'sea_ice', 'track_id'])
        ds = ds.assign_coords(Time=('Time', ds.Time))
        ds = ds.assign_coords(Latitude=ds.Latitude)
        ds = ds.assign_coords(Longitude=ds.Longitude)

        ds.Time.attrs['long_name'] = 'Time'
        ds.Time.attrs['standard_name'] = 'Time'
        adjusted_times = [reference_date + timedelta(seconds=time) for time in ds.Time.values]
        ds = ds.assign_coords(Time=adjusted_times)

        data_start_time = min(
            data_start_time, ds.Time.values[0]) if data_start_time else ds.Time.values[0]
        data_end_time = max(
            data_end_time, ds.Time.values[-1]) if data_end_time else ds.Time.values[-1]

        granules.append(ds)

    # Merge opened granules if needed
    cycle_ds = xr.concat((granules), dim='Time') if len(granules) > 1 else granules[0]

    # Time bounds

    # Center time
    data_center_time = data_start_time + ((data_end_time - data_start_time)/2)

    # Var Attributes
    cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(cycle_ds['SSHA'].values)
    cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(cycle_ds['SSHA'].values)

    # Time Attributes
    cycle_ds.Time.attrs['long_name'] = 'Time'

    # Global Attributes
    cycle_ds.attrs = {
        'title': 'Sea Level Anormaly Estimate based on Altimeter Data',
        'cycle_start': dates[0],
        'cycle_center': dates[1],
        'cycle_end': dates[2],
        'data_time_start': str(data_start_time)[:19],
        'data_time_center': str(data_center_time)[:19],
        'data_time_end': str(data_end_time)[:19],
        'original_dataset_title': ds_meta['original_dataset_title_s'],
        'original_dataset_short_name': ds_meta['original_dataset_short_name_s'],
        'original_dataset_url': ds_meta['original_dataset_url_s'],
        'original_dataset_reference': ds_meta['original_dataset_reference_s']
    }

    return cycle_ds, len(granules)


def process_measures_grids(cycle_granules, ds_meta, dates):
    """
    Processes and aggregates individual granules that fall within a cycle's date range for
    measures grids datasets (1812).

    Params:
        cycle_granules (List[dict]): the dataset specific config file
        ds_meta (dict): the list of docs to update on Solr
        dates (Tuple[str, str, str]):

    Returns:
        cycle_ds (Dataset): the processed cycle Dataset object
        1 (int): the number of granules within the processed cycle Dataset object
    """
    var = 'SLA'
    granule = cycle_granules[0]

    ds = xr.open_dataset(granule['granule_file_path_s'])

    if np.isnan(ds[var].values).all():
        granule = cycle_granules[1]
        ds = xr.open_dataset(granule['granule_file_path_s'])

    cycle_ds = ds.rename({var: 'SSHA'})

    # Var Attributes
    cycle_ds['SSHA'].attrs['valid_min'] = np.nanmin(cycle_ds['SSHA'].values)
    cycle_ds['SSHA'].attrs['valid_max'] = np.nanmax(cycle_ds['SSHA'].values)

    data_time_start = cycle_ds.Time_bounds.values[0][0]
    data_time_end = cycle_ds.Time_bounds.values[-1][1]
    data_time_center = data_time_start + ((data_time_end - data_time_start)/2)

    # Global attributes
    cycle_ds.attrs = {
        'title': 'Sea Level Anormaly Estimate based on Altimeter Data',
        'cycle_start': dates[0],
        'cycle_center': dates[1],
        'cycle_end': dates[2],
        'data_time_start': np.datetime_as_string(data_time_start, unit='s'),
        'data_time_center': np.datetime_as_string(data_time_center, unit='s'),
        'data_time_end': np.datetime_as_string(data_time_end, unit='s'),
        'original_dataset_title': ds_meta['original_dataset_title_s'],
        'original_dataset_short_name': ds_meta['original_dataset_short_name_s'],
        'original_dataset_url': ds_meta['original_dataset_url_s'],
        'original_dataset_reference': ds_meta['original_dataset_reference_s']
    }

    return cycle_ds, 1


def collect_granules(ds_name, dates, date_strs, config):
    """
    Collects granules that fall within a cycle's date range.
    The measures gridded dataset (1812) needs to only select the single granule closest
    to the center datetime of the cycle.

    Params:
        ds_name (str): the name of the dataset
        dates (Tuple[datetime, datetime, datetime]): the start, center, and end of the cycle 
                                                     in datetime format
        date_strs (Tuple[str, str, str]): the start, center, and end of the cycle in string format
        config (dict): the dataset specific config file


    Returns:
        cycle_granules (List[dict]): the Solr docs that satisfy the query
    """
    solr_regex = '%Y-%m-%dT%H:%M:%SZ'
    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    # Find the granule with date closest to center of cycle
    # Uses special Solr query function to automatically return granules in proximal order
    if '1812' in ds_name:
        query_start = datetime.strftime(dates[0], solr_regex)
        query_end = datetime.strftime(dates[2], solr_regex)
        fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true',
              f'date_dt:[{query_start} TO {query_end}}}']
        boost_function = f'recip(abs(ms({date_strs[1]}Z,date_dt)),3.16e-11,1,1)'

        query_params = {'q': '*:*',
                        'fq': fq,
                        'bf': boost_function,
                        'defType': 'edismax',
                        'rows': 300000,
                        'sort': 'date_s asc'}

        url = f'{solr_host}{solr_collection_name}/select?'
        response = requests.get(url, params=query_params)
        cycle_granules = response.json()['response']['docs']

    # Get granules within start_date and end_date
    else:
        query_start = datetime.strftime(dates[0], solr_regex)
        query_end = datetime.strftime(dates[2], solr_regex)
        fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true',
              f'date_dt:[{query_start} TO {query_end}}}']

        cycle_granules = solr_query(config, fq)

    return cycle_granules


def check_updating(cycles, date_strs, cycle_granules, version):
    """
    Checks whether a cycle requires reprocessing based on three conditions:
    - If the prior processing attempt failed
    - If the prior processing version differs from the current processing version
    - If any of the granules within the cycle date range have been modified since
        the prior processing attempt
    If the cycle has not previously been processed, check_updating returns True.

    Params:
        cycles (dict): the existing cycles on Solr in dictionary format where the key is
                        the start date string
        date_strs (Tuple[str, str, str]): the start, center, and end of the cycle in string format
        cycle_granules (List[dict]): the granules that make up the cycle
        version (float): the processing version number as defined in the dataset's config file


    Returns:
        (bool): whether or not the cycle requires reprocessing
    """

    # Cycles dict uses the PODAAC date format (with a trailing 'Z')
    if date_strs[0] + 'Z' in cycles.keys():
        existing_cycle = cycles[date_strs[0] + 'Z']

        prior_time = existing_cycle['processing_time_dt']
        prior_success = existing_cycle['processing_success_b']
        prior_version = existing_cycle['processing_version_f']

        if not prior_success or prior_version != version:
            return True

        for granule in cycle_granules:
            if prior_time < granule['modified_time_dt']:
                return True

        return False

    return True


def cycle_ds_encoding(cycle_ds, ds_name, center_date):
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
                                     'calendar': 'gregorian',
                                     'shuffle': False}
            # To account for time bounds in 1812 dataset
            if '1812' in ds_name:
                units_time = datetime.strftime(center_date, "%Y-%m-%d %H:%M:%S")
                coord_encoding[coord]['units'] = f'days since {units_time}'

        if 'Lat' in coord or 'Lon' in coord:
            coord_encoding[coord] = {'_FillValue': None, 'dtype': 'float32'}

    encoding = {**coord_encoding, **var_encodings}
    return encoding


def post_process_solr_update(config, ds_metadata):
    """
    Determines processing status by number of failed and successful cycle documents on Solr.
    Updates dataset document on Solr with status message

    Params:
        config (dict): the dataset's config file
        ds_metadata (dict): the dataset metadata document from Solr

    Return:
        processing_status (str): overall processing status
    """
    ds_name = config['ds_name']

    processing_status = 'All cycles successfully processed'

    # Query for failed cycle documents
    fq = ['type_s:cycle', f'dataset_s:{ds_name}', 'processing_success_b:false']
    failed_processing = solr_query(config, fq)

    if failed_processing:
        # Query for successful cycle documents
        fq = ['type_s:cycle', f'dataset_s:{ds_name}', 'processing_success_b:true']
        successful_processing = solr_query(config, fq)

        processing_status = 'No cycles successfully processed (all failed or no granules to process)'

        if successful_processing:
            processing_status = f'{len(failed_processing)} cycles failed'

    ds_metadata['processing_status_s'] = {"set": processing_status}
    resp = solr_update(config, [ds_metadata])

    if resp.status_code == 200:
        print('Successfully updated Solr dataset document\n')
    else:
        print('Failed to update Solr dataset document\n')

    return processing_status


def processing(config, output_path, reprocess):
    """
    Generates encoding dictionary used for saving the cycle netCDF file.
    The measures gridded dataset (1812) has additional units encoding requirements.

    Params:
        config (dict): the dataset's config file
        output_path (Path): path to the pipeline's output directory
        reprocess (bool): denotes if all cycles should be reprocessed
    """

    ds_name = config['ds_name']
    version = config['version']
    processor = config['processor']
    index_type = config['index_type']
    date_regex = '%Y-%m-%dT%H:%M:%S'

    if '1812' in ds_name:
        CYCLE_LENGTH = 5
    else:
        CYCLE_LENGTH = 10

    # Query for dataset metadata
    try:
        ds_metadata = solr_query(config, ['type_s:dataset', f'dataset_s:{ds_name}'])[0]
    except:
        log.exception('Error while querying for dataset entry.')
        raise Exception('Cannot run processing if harvesting has not been run. Check Solr.')

    # Query for all existing cycles in Solr
    solr_cycles = solr_query(config, ['type_s:cycle', f'dataset_s:{ds_name}'])

    cycles = {cycle['start_date_dt']: cycle for cycle in solr_cycles}

    # Generate list of cycle date tuples (start, end)
    delta = timedelta(days=CYCLE_LENGTH)
    start_date = datetime.strptime('1992-01-01T00:00:00', date_regex)
    end_date = start_date + delta

    while True:
        # Make strings for cycle start, center, and end dates
        start_date_str = datetime.strftime(start_date, date_regex)
        end_date_str = datetime.strftime(end_date, date_regex)
        center_date = start_date + ((end_date - start_date)/2)
        center_date_str = datetime.strftime(center_date, date_regex)

        dates = (start_date, center_date, end_date)
        date_strs = (start_date_str, center_date_str, end_date_str)

        # End cycle processing if cycles are outside of dataset date range
        if start_date_str > ds_metadata['end_date_dt']:
            break

        # Move to next cycle date range if end of cycle is before start of dataset
        if end_date_str < ds_metadata['start_date_dt']:
            start_date = end_date
            end_date = start_date + delta
            continue

        # ======================================================
        # Collect granules within cycle
        # ======================================================

        cycle_granules = collect_granules(ds_name, dates, date_strs, config)

        # Skip cycle if no granules harvested
        if not cycle_granules:
            print(f'No granules for cycle {start_date_str} to {end_date_str}')
            start_date = end_date
            end_date = start_date + delta
            continue

        # ======================================================
        # Determine if cycle requires processing
        # ======================================================

        if reprocess or check_updating(cycles, date_strs, cycle_granules, version):
            processing_success = False
            print(f'Processing cycle {start_date_str} to {end_date_str}')

            funcs = {'measures_grids': process_measures_grids,
                     'along_track': process_along_track}

            # ======================================================
            # Process the cycle
            # ======================================================

            try:
                # Dataset specific processing of cycle
                cycle_ds, granule_count = funcs[processor](cycle_granules, ds_metadata, date_strs)

                # Create netcdf encoding for cycle
                encoding = cycle_ds_encoding(cycle_ds, ds_name, center_date)

                # Save to netcdf
                filename_time = datetime.strftime(center_date, '%Y%m%dT%H%M%S')
                filename = f'ssha_{filename_time}.nc'

                save_dir = output_path / ds_name / 'cycle_products'
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir / filename

                cycle_ds.to_netcdf(save_path, encoding=encoding)

                # Determine checksum and file size
                checksum = md5(save_path)
                file_size = save_path.stat().st_size
                processing_success = True

            except:
                log.exception(
                    f'\nError while processing cycle {date_strs[0][:10]} - {date_strs[2][:10]}')
                filename = ''
                save_path = ''
                checksum = ''
                file_size = 0
                granule_count = 0

            # Add or update Solr cycle
            item = {
                'type_s': 'cycle',
                'dataset_s': ds_name,
                'start_date_dt': start_date_str,
                'center_date_dt': center_date_str,
                'end_date_dt': end_date_str,
                'granules_in_cycle_i': granule_count,
                'filename_s': filename,
                'filepath_s': str(save_path),
                'checksum_s': checksum,
                'file_size_l': file_size,
                'processing_success_b': processing_success,
                'processing_time_dt': datetime.utcnow().strftime(date_regex),
                'processing_version_f': version,
                'index_type_s': index_type
            }

            if start_date_str + 'Z' in cycles.keys():
                item['id'] = cycles[start_date_str + 'Z']['id']

            resp = solr_update(config, [item])
            if resp.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give granule documents the id of the corresponding cycle document
                if processing_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = ['type_s:cycle', f'dataset_s:{ds_name}', f'filename_s:{filename}']
                        cycle_id = solr_query(config, fq)[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    resp = solr_update(config, cycle_granules)

            else:
                print('\tFailed to create Solr cycle documents')
        else:
            print(f'No updates for cycle {start_date_str} to {end_date_str}')

        start_date = end_date
        end_date = start_date + delta

    # ======================================================
    # Update dataset document with overall processing status
    # ======================================================
    return post_process_solr_update(config, ds_metadata)
