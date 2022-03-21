"""
"""

import logging
import logging.config
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import requests
import yaml
from webdav3.client import Client
from matplotlib import pyplot as plt
import numexpr

from harvester import harvester
from indicators import indicators
from cycle_gridding import cycle_gridding
import txt_engine
import upload_indicators

from plotting import plot_generation
from utils import solr_utils
from conf.global_settings import OUTPUT_DIR

logging.config.fileConfig('logs/log.ini', disable_existing_loggers=False)
log = logging.getLogger(__name__)

# Set package logging level to WARNING
logs = ['requets', 'urllib3', 'webdav3',
        'matplotlib', 'numexpr', 'pyresample']
for l in logs:
    logging.getLogger(l).setLevel(logging.WARNING)

if not Path.is_dir(OUTPUT_DIR):
    print('Missing output directory. Please fill in. Exiting.')
    log.fatal('Missing output directory. Please fill in. Exiting.')
    exit()
print(f'\nUsing output directory: {OUTPUT_DIR}')

# Make sure Solr is up and running
try:
    solr_utils.ping_solr()
except requests.ConnectionError:
    print('Solr is not currently running! Start Solr and try again.')
    log.critical('Solr is not currently running! Start Solr and try again.')
    exit()

ds_status = defaultdict(list)

ROW = '=' * 57


def create_parser():
    """
    Creates command line argument parser

    Returns:
        parser (ArgumentParser): the ArgumentParser object
    """
    parser = ArgumentParser()

    parser.add_argument('--options_menu', default=False, action='store_true',
                        help='Display option menu to select which steps in the pipeline to run.')

    parser.add_argument('--force_processing', default=False, action='store_true',
                        help='Force reprocessing of any existing aggregated cycles.')

    parser.add_argument('--harvested_entry_validation', default=False,
                        help='verifies each Solr harvester entry points to a valid file.')

    return parser


def show_menu():
    """
    Prints the optional navigation menu

    Returns:
        selection (str): the menu number of the selection
    """
    while True:
        print(f'\n{" OPTIONS ":-^35}')
        print('1) Run pipeline on all')
        print('2) Harvest all datasets')
        print('3) Harvest single dataset')
        print('4) Perform gridding')
        print('5) Calculate index values and generate txt output and plots')
        print('6) Generate txt output and plots')
        print('7) Post indicators to FTP')
        selection = input('Enter option number: ')

        if selection in ['1', '2', '3', '4', '5', '6', '7']:
            return selection
        print(
            f'Unknown option entered, "{selection}", please enter a valid option\n')


def print_statuses():
    print('\n=========================================================')
    print(
        '=================== \033[36mPrinting statuses\033[0m ===================')
    print('=========================================================')
    for ds, status_list in ds_status.items():
        print(f'\033[93mPipeline status for {ds}\033[0m:')
        for msg in status_list:
            if 'success' in msg:
                print(f'\t\033[92m{msg}\033[0m')
            else:
                print(f'\t\033[91m{msg}\033[0m')


def run_harvester(datasets, configs, output_dir):
    """
        Calls the harvester with the dataset specific config file path for each
        dataset in datasets.

        Parameters:
            datasets (List[str]): A list of dataset names.
            output_dir (Path): The path to the output directory.
    """
    print(f'\n{ROW}')
    print(' \033[36mRunning harvesters\033[0m '.center(66, '='))
    print(f'{ROW}\n')

    for ds in datasets:
        try:
            print(f'\033[93mRunning harvester for {ds}\033[0m')
            print(ROW)
            ds_config = configs[ds]
            status = harvester(ds_config, output_dir)
            ds_status[ds].append(status)
            log.info(f'{ds} harvesting complete. {status}')
            print('\033[92mHarvest successful\033[0m')
        except Exception as e:
            ds_status[ds].append('Harvesting encountered error.')
            print(e)
            log.exception(f'{ds} harvesting failed. {e}')

            print('\033[91mHarvesting failed\033[0m')
        print(ROW)


def run_cycle_gridding(output_dir):
    print('\n' + ROW)
    print(' \033[36mRunning cycle gridding\033[0m '.center(66, '='))
    print(ROW + '\n')

    try:
        print(f'\033[93mRunning cycle gridding\033[0m')
        print(ROW)

        status = cycle_gridding(output_dir)
        ds_status['gridding'].append(status)
        log.info('Cycle gridding complete.')
        print('\033[92mCycle gridding complete\033[0m')
    except Exception as e:
        ds_status['gridding'].append('Gridding encountered error.')
        print(e)
        log.exception(f'Cycle gridding failed. {e}')
        print('\033[91mCycle gridding failed\033[0m')
    print(ROW)


def run_indexing(output_dir, reprocess):
    """
        Calls the indicator processing file.

        Parameters:
            output_dir (Path): The path to the output directory.
            reprocess: Boolean to force reprocessing
    """
    print('\n' + ROW)
    print(' \033[36mRunning index calculations\033[0m '.center(66, '='))
    print(ROW + '\n')

    success = False

    try:
        print('\033[93mRunning index calculation\033[0m')
        print(ROW)

        success = indicators(output_dir, reprocess)
        msg = 'Indicator calculation successful.'
        log.info('Index calculation complete.')
        print('\033[92mIndex calculation successful\033[0m')
    except Exception as e:
        print(e)
        msg = 'Indicator calculation encountered error.'
        log.error(f'Index calculation failed: {e}')
        print('\033[91mIndex calculation failed\033[0m')
    ds_status['indicators'].append(msg)
    print(ROW)

    return success


def run_post_processing(output_dir):
    plot_success = False
    try:
        plot_generation.main(output_dir)
        ds_status['plot_generation'].append(
            'Plot generation successfully completed.')
        plot_success = True
    except Exception as e:
        ds_status['plot_generation'].append(f'Plot generation failed: {e}')
        log.error(f'Plot generation failed: {e}')

    try:
        txt_engine.main(OUTPUT_DIR)
        log.info('Index txt file creation complete.')
        txt_success = True
    except Exception as e:
        log.error(f'Index txt file creation failed: {e}')


def post_to_ftp(OUTPUT_DIR):
    try:
        upload_indicators.main(OUTPUT_DIR)
        log.info('Index txt file upload complete.')
    except Exception as e:
        print(e)
        log.error(f'Index txt file upload failed: {e}')


if __name__ == '__main__':

    print('\n' + ROW)
    print(' SEA LEVEL INDICATORS PIPELINE '.center(57, '='))
    print(ROW)

    PARSER = create_parser()
    args = PARSER.parse_args()

    # -------------- Harvested Entry Validation --------------
    if args.harvested_entry_validation:
        solr_utils.validate_granules()

    # ------------------ Force Reprocessing ------------------
    REPROCESS = bool(args.force_processing)

    # --------------------- Run pipeline ---------------------

    with open(Path(f'conf/datasets.yaml'), "r") as stream:
        config = yaml.load(stream, yaml.Loader)
    configs = {c['ds_name']: c for c in config}

    DATASET_NAMES = list(configs.keys())

    CHOSEN_OPTION = show_menu() if args.options_menu and not REPROCESS else '1'

    # Run harvesting, gridding, indexing, post processing
    if CHOSEN_OPTION == '1':
        for dataset in DATASET_NAMES:
            run_harvester([dataset], configs, OUTPUT_DIR)
        run_cycle_gridding(OUTPUT_DIR)
        if run_indexing(OUTPUT_DIR, REPROCESS):
            run_post_processing()

    # Run all harvesters
    elif CHOSEN_OPTION == '2':
        for dataset in DATASET_NAMES:
            run_harvester([dataset], configs, OUTPUT_DIR)

    # Run specific harvester
    elif CHOSEN_OPTION == '3':
        ds_dict = dict(enumerate(DATASET_NAMES, start=1))
        while True:
            print('\nAvailable datasets:\n')
            for i, dataset in ds_dict.items():
                print(f'{i}) {dataset}')

            ds_index = input('\nEnter dataset number: ')

            if not ds_index.isdigit() or int(ds_index) not in range(1, len(DATASET_NAMES)+1):
                print(
                    f'Invalid dataset, "{ds_index}", please enter a valid selection')
            else:
                break

        CHOSEN_DS = ds_dict[int(ds_index)]
        print(f'\nHarvesting {CHOSEN_DS} dataset')

        run_harvester([CHOSEN_DS], configs, OUTPUT_DIR)

    # Run gridding
    elif CHOSEN_OPTION == '4':
        run_cycle_gridding(OUTPUT_DIR)

    # Run indexing (and post processing)
    elif CHOSEN_OPTION == '5':
        if run_indexing(OUTPUT_DIR, REPROCESS):
            run_post_processing(OUTPUT_DIR)

    # Run post processing
    elif CHOSEN_OPTION == '6':
        run_post_processing(OUTPUT_DIR)

    # Post txt file to website ftp
    elif CHOSEN_OPTION == '7':
        confirm = input(
            '\nPlease confirm posting indicators to ftp (y to confirm): ')

        if confirm != 'y':
            print('Not posting to ftp.')
        else:
            post_to_ftp(OUTPUT_DIR)

    print_statuses()
