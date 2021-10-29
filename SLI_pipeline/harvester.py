"""
This module handles data granule harvesting for datasets hosted locally and on PODAAC.
"""

import logging
import logging.config
import shutil
from datetime import datetime
from pathlib import Path
from xml.etree.ElementTree import fromstring

import requests
import yaml
from webdav3.client import Client
from utils import file_utils, solr_utils

logs_path = 'SLI_pipeline/logs/'
logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)


def print_resp(resp, msg=''):
    """
    Prints Solr response message

    Params:
        resp (Response): the response object from a solr update
        msg (str): the specific message to print
    """
    if resp.status_code == 200:
        print(f'Successfully created or updated Solr {msg}')
    else:
        print(f'Failed to create or update Solr {msg}')


def podaac_harvester(config, docs, target_dir):
    """
    Harvests new or updated granules from PODAAC for a specific dataset, within a
    specific date range. Creates new or modifies granule docs for each harvested granule.

    Params:
        config (dict): the dataset specific config file
        docs (dict): the existing granule docs on Solr in dict format
        target_dir (Path): the path of the dataset's harvested granules directory

    Returns:
        entries_for_solr (List[dict]): all new or modified granule docs to be posted to Solr
        url_base (str): PODAAC url for the specific dataset
    """

    ds_name = config['ds_name']
    shortname = config['original_dataset_short_name']

    now = datetime.utcnow()
    date_regex = config['date_regex']
    start_time = config['start']
    end_time = now.strftime(
        "%Y%m%dT%H:%M:%SZ") if config['most_recent'] else config['end']

    entries_for_solr = []

    if config['podaac_id']:
        url_base = f'{config["host"]}&datasetId={config["podaac_id"]}'
    else:
        url_base = f'{config["host"]}&shortName={shortname}'

    url = f'{url_base}&endTime={end_time}&startTime={start_time}'

    namespace = {"podaac": "http://podaac.jpl.nasa.gov/opensearch/",
                 "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
                 "atom": "http://www.w3.org/2005/Atom",
                 "georss": "http://www.georss.org/georss",
                 "gml": "http://www.opengis.net/gml",
                 "dc": "http://purl.org/dc/terms/",
                 "time": "http://a9.com/-/opensearch/extensions/time/1.0/"}

    # =====================================================
    # PODAAC loop
    # =====================================================
    while True:
        print('Loading granule entries from PODAAC XML document...')

        xml = fromstring(requests.get(url).text)
        items = xml.findall('{%(atom)s}entry' % namespace)

        # Loop through all granules in XML returned from URL
        for elem in items:
            updating = False

            # Extract granule information from XML entry and attempt to download data file

            # Extract download link from XML entry
            link = elem.find(
                "{%(atom)s}link[@title='OPeNDAP URL']" % namespace).attrib['href'][:-5]

            filename = link.split("/")[-1]

            # Extract start date from XML entry
            date_start_str = elem.find(
                "{%(time)s}start" % namespace).text[:19] + 'Z'

            # Extract modified time of file on podaac
            mod = elem.find("{%(atom)s}updated" % namespace)
            mod_time_str = mod.text

            # Granule metadata used for Solr granule entries
            item = {
                'type_s': 'granule',
                'date_dt': date_start_str,
                'dataset_s': ds_name,
                'filename_s': filename,
                'source_s': link,
                'modified_time_dt': mod_time_str
            }

            if filename in docs.keys():
                item['id'] = docs[filename]['id']

            local_dir = (target_dir / date_start_str[:4])
            local_dir.mkdir(parents=True, exist_ok=True)
            local_fp = local_dir / filename

            # If granule doesn't exist or previously failed or has been updated since last harvest
            # or exists in Solr but doesn't exist where it should
            updating = (filename not in docs.keys()) or \
                (not docs[filename]['harvest_success_b']) or \
                (docs[filename]['download_time_dt'] <= mod_time_str) or \
                (not local_fp.exists())

            # If updating, download file if necessary
            if updating:
                try:
                    expected_size = int(requests.head(
                        link).headers.get('content-length', -1))

                    # Only redownloads if local file is out of town - doesn't waste
                    # time/bandwidth to redownload the same file just because there isn't
                    # a Solr entry. Most useful during development.
                    if not local_fp.exists() or mod_time_str > datetime.fromtimestamp(local_fp.stat().st_mtime).strftime(date_regex):
                        print(f' - Downloading {filename} to {local_fp}')

                        resp = requests.get(link)
                        open(local_fp, 'wb').write(resp.content)
                    else:
                        print(
                            f' - {filename} already downloaded, but not in Solr.')

                    # Create checksum for file
                    item['checksum_s'] = file_utils.md5(local_fp)
                    item['granule_file_path_s'] = str(local_fp)
                    item['file_size_l'] = local_fp.stat().st_size

                    # Make sure file properly downloaded by comparing sizes
                    if expected_size == item['file_size_l']:
                        item['harvest_success_b'] = True
                    else:
                        item['harvest_success_b'] = False

                except:
                    log.exception(
                        f'{ds_name} harvesting error! {filename} failed to download')

                    item['harvest_success_b'] = False
                    item['checksum_s'] = ''
                    item['granule_file_path_s'] = ''
                    item['file_size_l'] = 0

                item['download_time_dt'] = now.strftime(date_regex)
                entries_for_solr.append(item)

            else:
                print(
                    f' - {filename} already downloaded, and up to date in Solr.')

        # Check if more granules are available on next page
        next_page = xml.find("{%(atom)s}link[@rel='next']" % namespace)
        if next_page is None:
            print(f'\nDownloading {ds_name} complete\n')
            break

        url = next_page.attrib['href'] + '&itemsPerPage=10000'

    return entries_for_solr, url_base


def podaac_drive_harvester(config, docs, target_dir):
    """
    Harvests new or updated granules from PODAAC for a specific dataset, within a
    specific date range. Creates new or modifies granule docs for each harvested granule.

    Params:
        config (dict): the dataset specific config file
        docs (dict): the existing granule docs on Solr in dict format
        target_dir (Path): the path of the dataset's harvested granules directory

    Returns:
        entries_for_solr (List[dict]): all new or modified granule docs to be posted to Solr
        source (str): PODAAC Restricted Drive url for the specific dataset
    """
    ds_name = config['ds_name']
    webdav_ds_name = ds_name.lower().replace('_', '-') + '/'

    now = datetime.utcnow()
    date_regex = config['date_regex']
    start_time = config['start']
    end_time = now.strftime(
        "%Y%m%dT%H:%M:%SZ") if config['most_recent'] else config['end']

    entries_for_solr = []

    with open(Path(f'{Path(__file__).resolve().parent}/login.yaml'), "r") as stream:
        earthdata_login = yaml.load(stream, yaml.Loader)

    webdav_options = {
        'webdav_hostname': 'https://podaac-tools.jpl.nasa.gov/drive-r/files/merged_alt/shared/L2/int',
        'webdav_login': earthdata_login['ed_user'],
        'webdav_password': earthdata_login['ed_password'],
        'disable_check': True
    }

    try:
        client = Client(webdav_options)
        print('Successfully connected to r-drive.')
    except Exception as e:
        log.error(f'Unable to login to r-drive. {e}')
        raise(f'Unable to login to r-drive. {e}')

    ds_years = client.list(webdav_ds_name)[1:]

    def date_filter(f):
        date = f['path'].split('_')[-1].split('.')[0][3:]
        start = start_time[:8]
        end = end_time[:8]
        if date >= start and date <= end:
            return True
        else:
            return False

    for year in ds_years:
        files = client.list(f'{webdav_ds_name}/{year}', get_info=True)[1:]
        files = [f for f in files if 'md5' not in f['path']]
        files.sort()
        files = filter(date_filter, files)

        for f in files:
            updating = False

            pod_name = f['path'].split('/')[-1]
            filename = f['path'].split('_')[-1]
            f_path = f'{webdav_ds_name}{year}{pod_name}'
            source = webdav_options['webdav_hostname'] + '/' + f_path

            date = filename.split('.')[0][3:]
            date_start_str = f'{date[:4]}-{date[4:6]}-{date[6:8]}T00:00:00Z'

            datetime_modified = datetime.strptime(
                f['modified'], '%a, %d %b %Y %H:%M:%S %Z')
            mod_time_str = datetime.strftime(
                datetime_modified, '%Y-%m-%dT%H:%M:%SZ')

            # Granule metadata used for Solr granule entries
            item = {
                'type_s': 'granule',
                'date_dt': date_start_str,
                'dataset_s': ds_name,
                'filename_s': filename,
                'source_s': source,
                'modified_time_dt': mod_time_str
            }

            if filename in docs.keys():
                item['id'] = docs[filename]['id']

            local_dir = (target_dir / date_start_str[:4])
            local_dir.mkdir(parents=True, exist_ok=True)
            local_fp = local_dir / filename

            updating = (filename not in docs.keys()) or \
                (not docs[filename]['harvest_success_b']) or \
                (docs[filename]['download_time_dt'] <= mod_time_str) or \
                (not local_fp.exists())

            if updating:
                try:
                    expected_size = f['size']

                    # Only redownloads if local file is out of town - doesn't waste
                    # time/bandwidth to redownload the same file just because there isn't
                    # a Solr entry. Most useful during development.
                    if not local_fp.exists() or mod_time_str > datetime.fromtimestamp(local_fp.stat().st_mtime).strftime(date_regex):
                        print(f' - Downloading {filename} to {local_fp}')

                        client.download_sync(remote_path=f_path,
                                             local_path=local_fp)
                    else:
                        print(
                            f' - {filename} already downloaded, but not in Solr.')

                    # Create checksum for file
                    item['checksum_s'] = file_utils.md5(local_fp)
                    item['granule_file_path_s'] = str(local_fp)
                    item['file_size_l'] = local_fp.stat().st_size

                    # Make sure file properly downloaded by comparing sizes
                    if int(expected_size) == item['file_size_l']:
                        item['harvest_success_b'] = True
                    else:
                        item['harvest_success_b'] = False

                except:
                    log.exception(
                        f'{ds_name} harvesting error! {filename} failed to download')

                    item['harvest_success_b'] = False
                    item['checksum_s'] = ''
                    item['granule_file_path_s'] = ''
                    item['file_size_l'] = 0

                item['download_time_dt'] = now.strftime(date_regex)
                entries_for_solr.append(item)

            else:
                print(
                    f' - {filename} already downloaded, and up to date in Solr.')

    return entries_for_solr, f'{webdav_options["webdav_hostname"]}/{webdav_ds_name}'


def local_harvester(config, docs, target_dir):
    """
    Harvests new or updated granules from a local drive for a specific dataset, within a
    specific date range. Creates new or modifies granule docs for each harvested granule.

    NOTE: Assumes data files are in expected directory structure:
    - SLI_output
        - {ds_name}
            - harvested_granules
                -{year}
                    - {ds_name}_YYYYMMDDTHHMMSSZ

    Params:
        config (dict): the dataset specific config file
        docs (dict): the existing granule docs on Solr in dict format
        target_dir (Path): the path of the dataset's harvested granules directory
    Returns:
        entries_for_solr (List[dict]): all new or modified granule metadata docs to be posted to Solr
        source (str): denotes granule/dataset was harvested from a local directory
    """
    ds_name = config['ds_name']
    date_regex = config['date_regex']
    source = 'Locally stored file'
    now = datetime.utcnow()
    now_str = now.strftime(date_regex)

    start_time = config['start']
    end_time = now.strftime(
        "%Y%m%dT%H:%M:%SZ") if config['most_recent'] else config['end']

    entries_for_solr = []
    print(target_dir)
    data_files = [filepath for filepath in target_dir.rglob("*.nc")]
    data_files.sort()

    print(data_files)
    exit()

    for filepath in data_files:

        date = filepath.name.split('_')[-1]
        date_start_str = f'{date[:4]}-{date[4:6]}-{date[6:11]}:{date[11:13]}:{date[13:16]}'

        mod_time = datetime.fromtimestamp(filepath.stat().st_mtime)
        mod_time_string = mod_time.strftime(date_regex)

        filename = filepath.name

        # Granule metadata used for Solr granule entries
        item = {
            'type_s': 'granule',
            'date_dt': date_start_str,
            'dataset_s': ds_name,
            'filename_s': filename,
            'source_s': source,
            'modified_time_dt': mod_time_string
        }

        if filename in docs.keys():
            item['id'] = docs[filename]['id']

        # If granule doesn't exist or previously failed or has been updated since last harvest
        updating = (filename not in docs.keys()) or \
            (not docs[filename]['harvest_success_b']) or \
            (docs[filename]['download_time_dt'] <= mod_time_string)

        if updating:
            print(f' - Adding {filename} to Solr.')

            # Create checksum for file
            item['checksum_s'] = md5(filepath)
            item['granule_file_path_s'] = str(filepath)
            item['harvest_success_b'] = True
            item['file_size_l'] = filepath.stat().st_size
            item['download_time_dt'] = now_str

            entries_for_solr.append(item)

        else:
            print(f' - {filename} already up to date in Solr.')

    return entries_for_solr, source


def harvester(config, output_path):
    """
    Harvests new or updated granules from a local drive for a dataset. Posts granule metadata docs
    to Solr and creates or updates dataset metadata doc.
    dataset doc.

    Params:
        config (dict): the dataset specific config file
        output_path (Path): the existing granule docs on Solr in dict format
    """

    # =====================================================
    # Setup variables from harvester_config.yaml
    # =====================================================
    ds_name = config['ds_name']
    shortname = config['original_dataset_short_name']

    target_dir = output_path / ds_name / 'harvested_granules'
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f'Harvesting {ds_name} files to {target_dir}\n')

    # =====================================================
    # Pull existing entries from Solr
    # =====================================================

    # Query for existing granule docs
    fq = ['type_s:granule', f'dataset_s:{ds_name}']
    harvested_docs = solr_utils.solr_query(fq)

    # Dictionary of existing granule docs
    # granule filename : solr entry for that doc
    docs = {}
    if harvested_docs:
        docs = {doc['filename_s']: doc for doc in harvested_docs}

    now_str = datetime.utcnow().strftime(config['date_regex'])

    # Actual downloading and generation of granule docs for Solr
    if config['harvester_type'] == 'podaac':
        entries_for_solr, source = podaac_harvester(config, docs, target_dir)
    elif config['harvester_type'] == 'PODAAC Drive':
        entries_for_solr, source = podaac_drive_harvester(
            config, docs, target_dir)
    elif config['harvester_type'] == 'local':
        entries_for_solr, source = local_harvester(config, docs, target_dir)

    # Only update Solr harvested entries if there are fresh downloads
    if entries_for_solr:
        # Update Solr with downloaded granule metadata entries
        resp = solr_utils.solr_update(entries_for_solr, True)
        print_resp(resp, msg='granule documents')

    # =====================================================
    # Solr dataset entry
    # =====================================================

    # Query for Solr failed harvest documents
    fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:false']
    failed_harvesting = solr_utils.solr_query(fq)

    # Query for Solr successful harvest documents
    fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true']
    successful_harvesting = solr_utils.solr_query(fq, sort='date_dt asc')

    if not successful_harvesting:
        harvest_status = 'No usable granules harvested (either all failed or no data collected)'
    elif failed_harvesting:
        harvest_status = f'{len(failed_harvesting)} harvested granules failed'
    else:
        harvest_status = 'All granules successfully harvested'

    # Query for Solr Dataset-level Document
    fq = ['type_s:dataset', f'dataset_s:{ds_name}']
    dataset_query = solr_utils.solr_query(fq)

    ds_start = successful_harvesting[0]['date_dt'] if successful_harvesting else None
    ds_end = successful_harvesting[-1]['date_dt'] if successful_harvesting else None

    # Query for Solr successful harvest documents
    fq = ['type_s:granule', f'dataset_s:{ds_name}', 'harvest_success_b:true']
    successful_harvesting = solr_utils.solr_query(
        fq, sort='download_time_dt desc')

    last_dl = successful_harvesting[0]['download_time_dt'] if successful_harvesting else None

    # -----------------------------------------------------
    # Create Solr dataset entry
    # -----------------------------------------------------

    ds_meta = {
        'type_s': 'dataset',
        'dataset_s': ds_name,
        'start_date_dt': ds_start,
        'end_date_dt': ds_end,
        'short_name_s': shortname,
        'source_s': source,
        'last_checked_dt': now_str,
        'last_download_dt': last_dl,
        'harvest_status_s': harvest_status,
        'original_dataset_title_s': config['original_dataset_title'],
        'original_dataset_short_name_s': shortname,
        'original_dataset_url_s': config['original_dataset_url'],
        'original_dataset_reference_s': config['original_dataset_reference'],
        'original_dataset_doi_s': config['original_dataset_doi']
    }

    if dataset_query:
        ds_meta['id'] = dataset_query[0]['id']

    # Update Solr with modified dataset entry
    resp = solr_utils.solr_update([ds_meta], True)
    print_resp(resp, msg='dataset document\n')

    return harvest_status
