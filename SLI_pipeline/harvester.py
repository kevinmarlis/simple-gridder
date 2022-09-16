import logging
import os
from datetime import datetime
from glob import glob
from pathlib import Path

import yaml
from webdav3.client import Client

from conf.global_settings import FILE_FORMAT


def drive_connection() -> Client:
    with open(Path(f'conf/login.yaml'), "r") as stream:
        earthdata_login = yaml.load(stream, yaml.Loader)

    webdav_options = {
        'webdav_hostname': 'https://podaac-tools.jpl.nasa.gov/drive-r/files/merged_alt/shared/L2/int',
        'webdav_login': earthdata_login['ed_user'],
        'webdav_password': earthdata_login['ed_password'],
        'disable_check': True
    }

    try:
        client = Client(webdav_options)
        logging.info('Successfully connected to r-drive.')
    except Exception as e:
        logging.error(f'Unable to login to r-drive. {e}')
        raise(f'Unable to login to r-drive. {e}')

    return client

def podaac_drive_harvester(config: dict, target_dir: Path) -> dict:
    """
    Harvests new or updated granules from PODAAC for a specific dataset, within a
    specific date range. Creates new or modifies granule docs for each harvested granule.

    Params:
        config (dict): the dataset specific config file
        target_dir (Path): the path of the dataset's harvested granules directory

    Returns:
        stats (dict): Dictionary containing harvesting statistics
    """
    client = drive_connection()

    ds_name = config['ds_name']
    if ds_name == 'MERGED_ALT':
        webdav_ds_name = ds_name.lower() + '/'
    else:
        webdav_ds_name = ds_name.lower().replace('_', '-') + '/'

    now = datetime.utcnow()
    start_time = config['start']
    start_year = start_time[:4]
    end_time = now.strftime("%Y%m%d") if config['end'] == 'now' else config['end']
    end_year = end_time[:4]

    years_range = list(range(int(start_year), int(end_year)+1))
    ds_years = client.list(webdav_ds_name)[1:]
    ds_years = [y for y in ds_years if int(y[:-1]) in years_range]
    ds_years.sort()

    def date_filter(f):
        date = f['path'].split('_')[-1].split('.')[0][3:]
        start = start_time
        end = end_time
        if date >= start and date <= end:
            return True
        return False

    stats = {'expected_files': 0}

    for year in ds_years:
        logging.info(f'Checking granules for {year}...This may take a while...')
        files = client.list(f'{webdav_ds_name}/{year}', get_info=True)[1:]
        files = [f for f in files if 'md5' not in f['path']]
        files.sort(key=lambda f: f['path'])
        files = list(filter(date_filter, files))
        stats['expected_files'] += len(files)


        local_dir = (target_dir / year)
        local_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            updating = False

            pod_name = f['path'].split('/')[-1]
            filename = f['path'].split('_')[-1]
            remote_path = f'{webdav_ds_name}{year}{pod_name}'
            
            datetime_modified = datetime.strptime(
                f['modified'], '%a, %d %b %Y %H:%M:%S %Z')

            local_fp = local_dir / filename

            if local_fp.exists():
                local_mod_time = datetime.fromtimestamp(os.path.getmtime(local_fp))
                local_size = local_fp.stat().st_size

            expected_size = f['size']

            updating = (not local_fp.exists()) or \
                (str(local_mod_time) <= str(datetime_modified)) or \
                    (int(expected_size) != local_size)

            if updating:
                logging.info(f'Downloading {ds_name} {filename}')
                try:
                    client.download_file(remote_path, local_fp)

                    local_size = local_fp.stat().st_size

                    # Make sure file properly downloaded by comparing sizes
                    if int(expected_size) != local_size:
                        logging.error(f'{ds_name} {filename} incomplete download. Removing file')
                        os.remove(local_fp)

                except Exception as e:
                    logging.info(f'{ds_name} harvesting error! {filename} failed to download')
                    logging.exception(e)

            else:
                logging.info(f'{ds_name} {filename} already up to date')

    stats['actual_files'] = len(glob(f'{target_dir}/**/*{FILE_FORMAT}', recursive=True))

    return stats

# https://podaac-tools.jpl.nasa.gov/drive-r/files/merged_alt/shared/L2/int/sentinel-6a/2020/SNTNL-6A-alt_ssh20201218.h5
# https://podaac-tools.jpl.nasa.gov/drive-r/files/merged_alt/shared/L2/int/merged_alt/1993/MERGED_ALT-alt_ssh19930101.h5
def harvester(config: dict, output_path: Path) -> str:
    """
    Harvests new or updated granules from a local drive for a dataset. Posts granule metadata docs
    to Solr and creates or updates dataset metadata doc.
    dataset doc.

    Params:
        config (dict): the dataset specific config file
        output_path (Path): the existing granule docs on Solr in dict format
    """
    ds_name = config['ds_name']

    target_dir = output_path / 'datasets' / ds_name / 'harvested_granules'
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f'Harvesting {ds_name} files to {target_dir}\n')

    stats = podaac_drive_harvester(config, target_dir)

    logging.info(f'{ds_name} harvesting: {stats["expected_files"]} expected files. {stats["actual_files"]} files harvested.')

    if stats['expected_files'] != stats['actual_files']:
        failed_count = stats['expected_files'] - stats['actual_files']
        harvest_status = f'{failed_count} harvested granules failed'
    else:
        harvest_status = 'All granules successfully harvested'

    return harvest_status
