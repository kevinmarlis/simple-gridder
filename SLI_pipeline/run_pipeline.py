"""
"""

import logging
import logging.config
from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import requests
import yaml
from webdav3.client import Client
from matplotlib import pyplot as plt
import h5py

from harvester import harvester
from indicators import indicators
from cycle_gridding import cycle_gridding
from plotting import plot_generation
from utils import solr_utils

RUN_TIME = datetime.now()


logs_path = Path('SLI_pipeline/logs/')
logs_path.mkdir(parents=True, exist_ok=True)

logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("webdav3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("h5py").setLevel(logging.WARNING)

# Hardcoded output directory path for pipeline files
OUTPUT_DIR = Path('/Users/marlis/Developer/SLI/sealevel_output/')
OUTPUT_DIR = Path('/Users/marlis/Developer/SLI/dev_output/')

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
    log.fatal('Solr is not currently running! Start Solr and try again.')
    # exit()

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
        print('3) Grid all datasets')
        print('4) Calculate index values')
        print('5) Dataset input')
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


def run_plot_generation(output_dir):
    plot_success = False
    try:
        plot_generation.main(output_dir)
        ds_status['plot_generation'].append(
            'Plot generation successfully completed.')
        plot_success = True
    except Exception as e:
        ds_status['plot_generation'].append(f'Plot generation failed: {e}')
        log.error(f'Plot generation failed: {e}')


def run_txt_gen_and_post(OUTPUT_DIR):
    import txt_engine
    import upload_indicators

    txt_success = False

    try:
        txt_engine.main(OUTPUT_DIR)
        log.info('Index txt file creation complete.')
        txt_success = True
    except Exception as e:
        log.error(f'Index txt file creation failed: {e}')

    if txt_success:
        try:
            upload_indicators.main(OUTPUT_DIR)
            log.info('Index txt file upload complete.')
        except Exception as e:
            log.error(f'Index txt file upload failed: {e}')
    else:
        log.error('Txt file creation failed. Not attempting to upload file.')


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

    config_path = Path(f'SLI_pipeline/configs/datasets.yaml')
    with open(config_path, "r") as stream:
        config = yaml.load(stream, yaml.Loader)
    configs = {c['ds_name']: c for c in config}

    DATASET_NAMES = list(configs.keys())

    CHOSEN_OPTION = show_menu() if args.options_menu and not REPROCESS else '1'
    # print('1) Run pipeline on all')
    # print('2) Harvest all datasets')
    # print('3) Grid all datasets')
    # print('4) Calculate index values')
    # print('5) Dataset input')

    # Run all
    if CHOSEN_OPTION == '1':
        for dataset in DATASET_NAMES:
            run_harvester([dataset], configs, OUTPUT_DIR)
        run_cycle_gridding(OUTPUT_DIR)
        if run_indexing(OUTPUT_DIR, REPROCESS):
            run_txt_gen_and_post()

    # Run harvesters
    elif CHOSEN_OPTION == '2':
        for dataset in DATASET_NAMES:
            run_harvester([dataset], configs, OUTPUT_DIR)

    # Run cycling
    elif CHOSEN_OPTION == '3':
        run_cycle_gridding(OUTPUT_DIR)

    elif CHOSEN_OPTION == '4':
        if run_indexing(OUTPUT_DIR, REPROCESS):
            run_plot_generation(OUTPUT_DIR)
            # run_txt_gen_and_post(OUTPUT_DIR)

    # Select dataset and pipeline step(s)
    elif CHOSEN_OPTION == '5':
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
        print(f'\nUsing {CHOSEN_DS} dataset')

        STEPS = ['harvest']
        steps_dict = dict(enumerate(STEPS, start=1))

        while True:
            print('\nAvailable steps:\n')
            for i, step in steps_dict.items():
                print(f'{i}) {step}')
            steps_index = input('\nEnter pipeline step(s) number: ')

            if not steps_index.isdigit() or int(steps_index) not in range(1, len(STEPS)+1):
                print(
                    f'Invalid step(s), "{steps_index}", please enter a valid selection')
            else:
                break

        wanted_steps = steps_dict[int(steps_index)]

        if 'harvest' in wanted_steps:
            run_harvester([CHOSEN_DS], configs, OUTPUT_DIR)
    elif CHOSEN_OPTION == '6':
        run_plot_generation(OUTPUT_DIR)
    elif CHOSEN_OPTION == '7':
        # run_plot_generation(OUTPUT_DIR)
        # run_txt_gen_and_post(OUTPUT_DIR)
        pass

    print_statuses()
