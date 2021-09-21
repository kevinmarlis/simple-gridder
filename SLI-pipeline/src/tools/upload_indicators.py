from paramiko import SSHClient
from scp import SCPClient
import yaml
import os

def main(output_dir):
    with open(f'{os.getcwd()}/SLI-pipeline/src/tools/ftp_login.yaml', "r") as stream:
        config = yaml.load(stream, yaml.Loader)

    data_path = output_dir / f'indicator/indicator_data.txt'
    upload_path = '/home/sealevel/ftp/indicator_data.txt'

    print('Uploading file to FTP')

    ssh = SSHClient()
    ssh.load_system_host_keys()

    try:
        ssh.connect(hostname=config['host'],
                    username=config['username'],
                    password=config['password'])
    except Exception as e:
        raise(f'FTP connection error. {e}')

    try:
        with SCPClient(ssh.get_transport()) as scp:
            scp.put(data_path, upload_path)
    except Exception as e:
        raise(f'Unable to upload file. {e}')