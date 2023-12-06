import logging
from argparse import ArgumentParser
from pathlib import Path

import yaml

from conf.global_settings import OUTPUT_DIR
from cycle_gridding import cycle_gridding
from logs.logconfig import configure_logging



def create_parser():
    """
    Creates command line argument parser

    Returns:
        parser (ArgumentParser): the ArgumentParser object
    """
    parser = ArgumentParser()

    parser.add_argument('satellite', choices=['SNTNL-6A', 'SNTNL-3B', 'SNTNL-3A', 'SARAL', 
                                              'MERGED_ALT', 'ERS-2', 'ERS-1', 'ENVISAT1', 'CRYOSAT2'],
                        help='Single dataset to perform simple gridding on')
    parser.add_argument('start', help='Start of date range to process. In format %Y-%m-%d')
    parser.add_argument('-end', default='now', required=False, help='Optional end of date range to process in format %Y-%m-%d. Default of \'now\'')
    parser.add_argument('--log_level',  default='INFO', choices = ['INFO', 'DEBUG', 'WARNING'], help='Sets the log level. Default of INFO')


    return parser

if __name__ == '__main__':

    PARSER = create_parser()
    args = PARSER.parse_args()

    configure_logging(file_timestamp=True, log_level=args.log_level)

    if not Path.is_dir(OUTPUT_DIR):
        logging.fatal('Output directory does not exist. Exiting.')
        exit()
    logging.debug(f'\nUsing output directory: {OUTPUT_DIR}')

    # --------------------- Run pipeline ---------------------

    with open(Path(f'conf/datasets.yaml'), "r") as stream:
        config = yaml.load(stream, yaml.Loader)
    configs = {c['ds_name']: c for c in config}

    try:
        cycle_gridding(args.satellite, args.start, args.end)
        logging.info('Cycle gridding complete.')
    except Exception as e:
        logging.exception(f'Cycle gridding failed. {e}')
    