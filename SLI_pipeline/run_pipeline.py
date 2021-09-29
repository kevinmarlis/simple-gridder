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

from cycle_creation import cycle_creation
from harvester import harvester
import indicators
from regridding import regridding
from utils import solr_utils

RUN_TIME = datetime.now()


logs_path = Path('SLI_pipeline/logs/')
logs_path.mkdir(parents=True, exist_ok=True)

logging.config.fileConfig(f'{logs_path}/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Hardcoded output directory path for pipeline files
# Leave blank to be prompted for an output directory
OUTPUT_DIR = Path('/Users/marlis/Developer/SLI/sealevel_output/')
if not Path.is_dir(OUTPUT_DIR):
    log.fatal('Missing output directory. Please fill in. Exiting.')
    exit()
print(f'\nUsing output directory: {OUTPUT_DIR}')

# Make sure Solr is up and running
try:
    solr_utils.ping_solr()
except requests.ConnectionError:
    log.fatal('Solr is not currently running! Start Solr and try again.')
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
        print('3) Update cycles for all datasets')
        print('4) Regrid DAILY and MEASURES datasets')
        print('5) Dataset input')
        print('6) Calculate index values')
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


def run_harvester(datasets, output_dir):
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

            config_path = Path(f'SLI_pipeline/dataset_configs/{ds}.yaml')
            with open(config_path, "r") as stream:
                config = yaml.load(stream, yaml.Loader)

            status = harvester(config, output_dir)
            ds_status[ds].append(status)
            log.info(f'{ds} harvesting complete. {status}')
            print('\033[92mHarvest successful\033[0m')
        except Exception as e:
            ds_status[ds].append('Harvesting encountered error.')
            print(e)
            log.exception(f'{ds} harvesting failed. {e}')

            print('\033[91mHarvesting failed\033[0m')
        print(ROW)


def run_cycle_creation(datasets, output_dir, reprocess):
    """
        Calls the processor with the dataset specific config file path for each
        dataset in datasets.

        Parameters:
            datasets (List[str]): A list of dataset names.
            output_dir (Pathh): The path to the output directory.
            reprocess: Boolean to force reprocessing
    """
    print('\n' + ROW)
    print(' \033[36mRunning cycle creation\033[0m '.center(66, '='))
    print(ROW + '\n')

    for ds in datasets:
        try:
            print(f'\033[93mRunning cycle creation for {ds}\033[0m')
            print(ROW)

            config_path = Path(f'SLI_pipeline/dataset_configs/{ds}.yaml')
            with open(config_path, "r") as stream:
                config = yaml.load(stream, yaml.Loader)

            status = cycle_creation(config, output_dir, reprocess)
            ds_status[ds].append(status)
            log.info(f'{ds} cycle creation complete. {status}')
            print('\033[92mCycle creation complete\033[0m')
        except Exception as e:
            ds_status[ds].append('Cycle creation encountered error.')
            print(e)
            log.exception(f'{ds} cycle creation failed. {e}')
            print('\033[91mCycle creation failed\033[0m')
        print(ROW)


def run_cycle_regridding(output_dir, reprocess):
    """
        Calls the processor with the dataset specific config file path for each
        dataset in datasets.

        Parameters:
            datasets (List[str]): A list of dataset names.
            output_dir (Pathh): The path to the output directory.
            reprocess: Boolean to force reprocessing
    """
    print('\n' + ROW)
    print(' \033[36mRunning cycle regridding\033[0m '.center(66, '='))
    print(ROW + '\n')

    try:
        print(f'\033[93mRunning cycle regridding\033[0m')
        print(ROW)

        status = regridding(output_dir, reprocess)
        ds_status['regridding'].append(status)
        log.info('Cycle regridding complete.')
        print('\033[92mCycle regridding complete\033[0m')
    except Exception as e:
        ds_status['regridding'].append('Regridding encountered error.')
        print(e)
        log.exception(f'Cycle regridding failed. {e}')
        print('\033[91mCycle regridding failed\033[0m')
    print(ROW)


def run_indexing(src_path, output_dir, reprocess):
    """
        Calls the indicator processing file.

        Parameters:
            src_path (Path): The path to the src directory.
            output_dir (Path): The path to the output directory.
            reprocess: Boolean to force reprocessing
    """
    print('\n' + ROW)
    print(' \033[36mRunning index calculations\033[0m '.center(66, '='))
    print(ROW + '\n')

    updated = False

    try:
        print('\033[93mRunning index calculation\033[0m')
        print(ROW)

        config_path = Path(
            src_path/'processors/indicators/indicators_config.yaml')
        with open(config_path, "r") as stream:
            config = yaml.load(stream, yaml.Loader)

        updated = indicators(config, output_dir, reprocess)

        log.info('Index calculation complete.')
        print('\033[92mIndex calculation successful\033[0m')
    except Exception as e:
        print(e)
        log.error(f'Index calculation failed: {e}')
        print('\033[91mIndex calculation failed\033[0m')
    print(ROW)

    return updated


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

    DATASETS = [ds.name.replace('.yaml', '') for ds in Path(
        'SLI_pipeline/dataset_configs').iterdir() if '.yaml' in ds.name]

    DATASETS.sort()

    CHOSEN_OPTION = show_menu() if args.options_menu and not REPROCESS else '1'
    # print('1) Run pipeline on all')
    # print('2) Harvest all datasets')
    # print('3) Update cycles for all datasets')
    # print('4) Regrid DAILY and MEASURES datasets')
    # print('5) Dataset input')
    # print('6) Calculate index values')

    # Run all
    if CHOSEN_OPTION == '1':
        for dataset in DATASETS:
            run_harvester([dataset], OUTPUT_DIR)
            run_cycle_creation([dataset], OUTPUT_DIR, REPROCESS)
        run_cycle_regridding(OUTPUT_DIR, REPROCESS)
        if run_indexing(OUTPUT_DIR, REPROCESS):
            run_txt_gen_and_post()

    # Run harvesters
    elif CHOSEN_OPTION == '2':
        for dataset in DATASETS:
            run_harvester([dataset], OUTPUT_DIR)

    # Run cycling
    elif CHOSEN_OPTION == '3':
        for dataset in DATASETS:
            run_cycle_creation([dataset], OUTPUT_DIR, REPROCESS)

    # Run cycle regridding
    elif CHOSEN_OPTION == '4':
        run_cycle_regridding(OUTPUT_DIR, REPROCESS)

    # Select dataset and pipeline step(s)
    elif CHOSEN_OPTION == '5':
        ds_dict = dict(enumerate(DATASETS, start=1))
        while True:
            print('\nAvailable datasets:\n')
            for i, dataset in ds_dict.items():
                print(f'{i}) {dataset}')

            ds_index = input('\nEnter dataset number: ')

            if not ds_index.isdigit() or int(ds_index) not in range(1, len(DATASETS)+1):
                print(
                    f'Invalid dataset, "{ds_index}", please enter a valid selection')
            else:
                break

        CHOSEN_DS = ds_dict[int(ds_index)]
        print(f'\nUsing {CHOSEN_DS} dataset')

        STEPS = ['harvest', 'create cycles', 'regrid cycles', 'all']
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
            run_harvester([CHOSEN_DS], OUTPUT_DIR)
        if 'create' in wanted_steps:
            run_cycle_creation([CHOSEN_DS], OUTPUT_DIR, REPROCESS)
        if 'regrid' in wanted_steps:
            run_cycle_regridding(OUTPUT_DIR, REPROCESS)
            # run_indexing(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)
        if wanted_steps == 'all':
            run_harvester([CHOSEN_DS], OUTPUT_DIR)
            run_cycle_creation([CHOSEN_DS], OUTPUT_DIR, REPROCESS)
            run_cycle_regridding(OUTPUT_DIR, REPROCESS)
            # run_indexing(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)

    elif CHOSEN_OPTION == '6':
        if run_indexing(OUTPUT_DIR, REPROCESS):
            run_txt_gen_and_post(OUTPUT_DIR)

    elif CHOSEN_OPTION == '7':
        run_txt_gen_and_post(OUTPUT_DIR)

    print_statuses()
