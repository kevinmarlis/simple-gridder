"""
This module handles dataset processing into 5, 10, or 30 day cycles.
"""
import logging
import warnings
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from netCDF4 import default_fillvals  # pylint: disable=no-name-in-module

from utils import file_utils, solr_utils

# warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")


logs_path = 'SLI_pipeline/logs/'
logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)


def process_along_track(cycle_granules, ds_meta, dates, CYCLE_LENGTH):
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
    granules = []

    for granule in cycle_granules:
        # along track data is in hdf5 format and uses groups
        try:
            ds = xr.open_dataset(granule['granule_file_path_s'], group='data')
            ds = ds.rename(
                {'lats': 'latitude', 'lons': 'longitude', 'ssh': 'SSHA'})
            ds = ds[['SSHA', 'latitude', 'longitude', 'time']]
            dim = ds.SSHA.dims[0]
            ds = ds.swap_dims({dim: 'time'})

            ds.time.attrs = {
                'long_name': 'time',
                'standard_name': 'time',
                'units': 'seconds since 1985-01-01',
                'calendar': 'gregorian',
            }

        # GSFC data is in netcdf format and doesn't use groups
        except:
            ds = xr.open_dataset(granule['granule_file_path_s'])

            if 'ssha' in ds.data_vars:
                var = 'ssha'
            elif 'sla' in ds.data_vars:
                var = 'sla'
            elif 'sla_mle3' in ds.data_vars:
                var = 'sla_mle3'

            ds = ds.rename(
                {'lat': 'latitude', 'lon': 'longitude', var: 'SSHA'})

            if 'time_rel_eq' in ds.data_vars:
                eq_dt = datetime.strptime(
                    ds.attrs['equator_time'], '%Y-%m-%d %H:%M:%S.%f')

                adjusted_times = [eq_dt + timedelta(seconds=time)
                                  for time in ds.time_rel_eq.values]
                ds = ds.assign_coords(time=('time', adjusted_times))

                ds.time.attrs = {
                    'long_name': 'time',
                    'standard_name': 'time',
                }

            if granule['dataset_s'] == 'GSFC':
                ds = ds.swap_dims({'N_Records': 'time'})

                #  Convert to meters
                ds['SSHA'] = ds['SSHA'] / 1e3

            ds = ds[['SSHA', 'latitude', 'longitude', 'time']]

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

        granules.append(ds)

    print('\tMerging granules...')
    # Merge opened granules if needed
    cycle_ds = xr.concat((granules), dim='time') if len(
        granules) > 1 else granules[0]

    # Global Attributes
    cycle_ds.attrs = {
        'title': 'Sea Surface Height Anormaly Estimate based on Altimeter Data',
        'comment': f'Data aggregated into {CYCLE_LENGTH} day cycle periods.',
        'cycle_period': f'{dates[0]} to {dates[2]}',
        'cycle_length': CYCLE_LENGTH,
        'cycle_start': dates[0],
        'cycle_center': dates[1],
        'cycle_end': dates[2],
        'original_dataset_title': ds_meta['original_dataset_title_s'],
        'original_dataset_short_name': ds_meta['original_dataset_short_name_s'],
        'original_dataset_url': ds_meta['original_dataset_url_s'],
        'original_dataset_reference': ds_meta['original_dataset_reference_s']
    }

    return cycle_ds, len(granules)


def process_measures_grids(cycle_granules, ds_meta, dates, CYCLE_LENGTH):
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

    if np.isnan(ds[var].values).all() and len(cycle_granules) > 1:
        granule = cycle_granules[1]
        ds = xr.open_dataset(granule['granule_file_path_s'])

    cycle_ds = ds.rename(
        {'Latitude': 'latitude', 'Longitude': 'longitude', 'SLA': 'SSHA', 'Time': 'time'})
    cycle_ds = cycle_ds.drop(['SLA_ERR'])

    cycle_ds.latitude.attrs = {
        'long_name': 'latitude',
        'standard_name': 'latitude',
        'units': 'degrees_north',
        'valid_min': np.nanmin(cycle_ds.latitude.values),
        'valid_max': np.nanmax(cycle_ds.latitude.values),
        'comment': 'Positive latitude is North latitude, negative latitude is South latitude. FillValue pads the reference orbits to have same length'
    }

    cycle_ds.longitude.attrs = {
        'long_name': 'longitude',
        'standard_name': 'longitude',
        'units': 'degrees_east',
        'valid_min': np.nanmin(cycle_ds.longitude.values),
        'valid_max': np.nanmax(cycle_ds.longitude.values),
        'comment': 'East longitude relative to Greenwich meridian. FillValue pads the reference orbits to have same length'
    }

    cycle_ds.SSHA.attrs = {
        'long_name': 'sea surface height anomaly',
        'standard_name': 'sea_surface_height_above_sea_level',
        'units': 'm',
        'valid_min': np.nanmin(cycle_ds.SSHA.values),
        'valid_max': np.nanmax(cycle_ds.SSHA.values),
        'comment': 'Sea level determined from satellite altitude - range - all altimetric corrections',
    }

    # Global attributes
    cycle_ds.attrs = {
        'title': 'Sea Surface Height Anormaly Estimate based on Altimeter Data',
        'comment': f'Data aggregated into {CYCLE_LENGTH} day cycle periods.',
        'cycle_period': f'{dates[0]} to {dates[2]}',
        'cycle_length': CYCLE_LENGTH,
        'cycle_start': dates[0],
        'cycle_center': dates[1],
        'cycle_end': dates[2],
        'original_dataset_title': ds_meta['original_dataset_title_s'],
        'original_dataset_short_name': ds_meta['original_dataset_short_name_s'],
        'original_dataset_url': ds_meta['original_dataset_url_s'],
        'original_dataset_reference': ds_meta['original_dataset_reference_s']
    }

    return cycle_ds, 1


def collect_granules(ds_name, dates, date_strs):
    """
    Collects granules that fall within a cycle's date range.
    The measures gridded dataset (1812) needs to only select the single granule closest
    to the center datetime of the cycle.

    Params:
        ds_name (str): the name of the dataset
        dates (Tuple[datetime, datetime, datetime]): the start, center, and end of the cycle 
                                                     in datetime format
        date_strs (Tuple[str, str, str]): the start, center, and end of the cycle in string format

    Returns:
        cycle_granules (List[dict]): the Solr docs that satisfy the query
    """
    solr_regex = '%Y-%m-%dT%H:%M:%SZ'
    query_start = datetime.strftime(dates[0], solr_regex)
    query_end = datetime.strftime(dates[2], solr_regex)
    fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true',
          f'date_dt:[{query_start} TO {query_end}}}']

    # Find the granule with date closest to center of cycle
    # Uses special Solr query function to automatically return granules in proximal order
    if '1812' in ds_name:
        boost_function = f'recip(abs(ms({date_strs[1]}Z,date_dt)),3.16e-11,1,1)'

        cycle_granules = solr_utils.solr_query_boost(fq, boost_function)

    # Get granules within start_date and end_date
    else:
        cycle_granules = solr_utils.solr_query(fq)

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
                                     'shuffle': False}
            # To account for time bounds in 1812 dataset
            if '1812' in ds_name:
                units_time = datetime.strftime(
                    center_date, "%Y-%m-%d %H:%M:%S")
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
    failed_processing = solr_utils.solr_query(fq)

    if failed_processing:
        # Query for successful cycle documents
        fq = ['type_s:cycle',
              f'dataset_s:{ds_name}', 'processing_success_b:true']
        successful_processing = solr_utils.solr_query(fq)

        processing_status = 'No cycles successfully processed (all failed or no granules to process)'

        if successful_processing:
            processing_status = f'{len(failed_processing)} cycles failed'

    ds_metadata['processing_status_s'] = {"set": processing_status}
    resp = solr_utils.solr_update([ds_metadata], True)

    if resp.status_code == 200:
        print('Successfully updated Solr dataset document\n')
    else:
        print('Failed to update Solr dataset document\n')

    return processing_status


def cycle_creation(config, output_path, reprocess):
    """
    Generates encoding dictionary used for saving the cycle netCDF file.
    The measures gridded dataset (1812) has additional units encoding requirements.

    Params:
        config (dict): the dataset's config file
        output_path (Path): path to the pipeline's output directory
        reprocess (bool): denotes if all cycles should be reprocessed
    """

    # =====================================================
    # Setup variables from config.yaml
    # =====================================================
    ds_name = config['ds_name']
    version = config['version']
    data_type = config['data_type']
    date_regex = '%Y-%m-%dT%H:%M:%S'

    CYCLE_LENGTH = config['cycle_length']

    # Query for dataset metadata
    try:
        fq = ['type_s:dataset', f'dataset_s:{ds_name}']
        ds_metadata = solr_utils.solr_query(fq)[0]
    except:
        log.exception(
            f'Error while querying for {ds_name} dataset entry. Cannot create cycles if harvesting has not been run. Check Solr.')
        raise Exception(
            'No granules have been harvested. Check Solr.')

    if 'end_date_dt' not in ds_metadata.keys():
        log.exception(f'No granules harvested for {ds_name}.')
        raise Exception(
            f'No granules harvested for {ds_name}. Check Solr.')

    # Query for all existing cycles in Solr
    fq = ['type_s:cycle', f'dataset_s:{ds_name}']
    solr_cycles = solr_utils.solr_query(fq)

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

        cycle_granules = collect_granules(ds_name, dates, date_strs)

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

            # ======================================================
            # Process the cycle
            # ======================================================

            try:
                # Dataset specific processing of cycle
                if data_type == 'along track':
                    cycle_ds, granule_count = process_along_track(cycle_granules,
                                                                  ds_metadata,
                                                                  date_strs,
                                                                  CYCLE_LENGTH)
                elif data_type == 'gridded':
                    cycle_ds, granule_count = process_measures_grids(cycle_granules,
                                                                     ds_metadata,
                                                                     date_strs,
                                                                     CYCLE_LENGTH)

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
                checksum = file_utils.md5(save_path)
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
                processing_success = False

            # Add or update Solr cycle
            item = {
                'type_s': 'cycle',
                'dataset_s': ds_name,
                'start_date_dt': start_date_str,
                'center_date_dt': center_date_str,
                'end_date_dt': end_date_str,
                'granules_in_cycle_i': granule_count,
                'cycle_length_i': CYCLE_LENGTH,
                'filename_s': filename,
                'filepath_s': str(save_path),
                'checksum_s': checksum,
                'file_size_l': file_size,
                'processing_success_b': processing_success,
                'processing_time_dt': datetime.utcnow().strftime(date_regex),
                'processing_version_f': version,
                'data_type_s': data_type
            }

            if start_date_str + 'Z' in cycles.keys():
                item['id'] = cycles[start_date_str + 'Z']['id']

            resp = solr_utils.solr_update([item], True)
            if resp.status_code == 200:
                print('\tSuccessfully created or updated Solr cycle documents')

                # Give granule documents the id of the corresponding cycle document
                if processing_success:
                    if 'id' in item.keys():
                        cycle_id = item['id']
                    else:
                        fq = ['type_s:cycle',
                              f'dataset_s:{ds_name}', f'filename_s:{filename}']
                        cycle_id = solr_utils.solr_query(fq)[0]['id']

                    for granule in cycle_granules:
                        granule['cycle_id_s'] = cycle_id

                    resp = solr_utils.solr_update(cycle_granules)

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
