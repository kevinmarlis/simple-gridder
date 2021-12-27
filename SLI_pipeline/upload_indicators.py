import logging
import logging.config
import os

import yaml
from paramiko import SSHClient
from scp import SCPClient

logs_path = 'SLI_pipeline/logs/'
logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)


def main(output_dir):
    with open(f'{os.getcwd()}/SLI_pipeline/configs/login.yaml', "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    data_path = output_dir / f'indicator/indicator_data.txt'
    upload_path = '/home/sealevel/ftp/indicator_data.txt'

    print('Uploading file to FTP')

    ssh = SSHClient()
    ssh.load_system_host_keys()

    try:
        ssh.connect(hostname=config['ftp_host'],
                    username=config['ftp_username'],
                    password=config['ftp_password'])
    except Exception as e:
        print(e)
        log.error(f'Failed to connect to FTP. {e}')
        raise(f'FTP connection error. {e}')

    try:
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(data_path, upload_path)
        print('Indicators successfully pushed to FTP')
        log.debug('Indicators successfully pushed to FTP')
    except Exception as e:
        print(e)
        log.error(f'Indicators failed to push to FTP. {e}')
        raise(f'Unable to upload file. {e}')
