"""
"""
import os
import sys
import logging
from argparse import ArgumentParser
import importlib
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from collections import defaultdict
import requests
import yaml
from datetime import datetime

# Hardcoded output directory path for pipeline files
# Leave blank to be prompted for an output directory
OUTPUT_DIR = ''
OUTPUT_DIR = Path('/Users/marlis/Developer/SLI/sealevel_output/')

LOG_TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

if OUTPUT_DIR:
    logs_path = Path(OUTPUT_DIR / f'logs/{LOG_TIME}/')
    logs_path.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# formatter = logging.Formatter('%(asctime)s: %(message)s')


ROW = '=' * 57


def create_parser():
    """
    Creates command line argument parser

    Returns:
        parser (ArgumentParser): the ArgumentParser object
    """
    parser = ArgumentParser()

    parser.add_argument('--output_dir', default=False, action='store_true',
                        help='Runs prompt to select pipeline output directory.')

    parser.add_argument('--options_menu', default=False, action='store_true',
                        help='Display option menu to select which steps in the pipeline to run.')

    parser.add_argument('--force_processing', default=False, action='store_true',
                        help='Force reprocessing of any existing aggregated cycles.')

    parser.add_argument('--harvested_entry_validation', default=False,
                        help='verifies each Solr harvester entry points to a valid file.')

    return parser


def verify_solr_running():
    """
    Verifies that Solr is up and running.
    Quits the pipeline if Solr can't be pinged.
    """
    solr_host = 'http://localhost:8983/solr/'
    solr_collection_name = 'sealevel_datasets'

    try:
        requests.get(f'{solr_host}{solr_collection_name}/admin/ping')
        return
    except requests.ConnectionError:
        print('\nSolr not currently running! Please double check and run pipeline again.\n')
        sys.exit()


def harvested_entry_validation():
    """Validates local file pointed to in granule entries"""
    solr_host = 'http://localhost:8983/solr/'
    solr_collection_name = 'sealevel_datasets'

    response = requests.get(
        f'{solr_host}{solr_collection_name}/select?fq=type_s%3Agranule&q=*%3A*')

    if response.status_code == 200:
        docs_to_remove = []
        harvested_docs = response.json()['response']['docs']

        for doc in harvested_docs:
            file_path = doc['pre_transformation_file_path_s']
            if os.path.exists(file_path):
                continue
            docs_to_remove.append(doc['id'])

        url = f'{solr_host}{solr_collection_name}/update?commit=true'
        requests.post(url, json={'delete': docs_to_remove})

        print('Succesfully removed entries from Solr')

    else:
        print('Solr not online or collection does not exist')
        sys.exit()


def show_menu():
    """
    Prints the optional navigation menu

    Returns:
        selection (str): the menu number of the selection
    """
    while True:
        print(f'\n{" OPTIONS ":-^35}')
        print('1) Harvest and process all datasets')
        print('2) Harvest all datasets')
        print('3) Update cycles for all datasets')
        print('4) Regrid cycles for all datasets')
        print('5) Dataset input')
        print('6) Calculate index values')
        selection = input('Enter option number: ')

        if selection in ['1', '2', '3', '4', '5', '6']:
            return selection
        print(
            f'Unknown option entered, "{selection}", please enter a valid option\n')


def print_log(output_dir):
    """
        Prints pipeline log summary.

        Parameters:
            output_dir (str): The path to the output directory.
    """

    print('\n' + ROW)
    print(' \033[36mPrinting log\033[0m '.center(66, '='))
    print(ROW)
    log_path = output_dir / f'logs/{LOG_TIME}/pipeline.log'
    dataset_statuses = defaultdict(lambda: defaultdict(list))
    index_statuses = []
    # Parse logger for messages
    with open(log_path) as log:
        logs = log.read().splitlines()

    for line in logs:
        log_line = yaml.load(line, yaml.Loader)

        if 'harvesting' in log_line['message']:
            ds = log_line['message'].split()[0]
            step = 'harvesting'
            msg = log_line['message'].replace(f'{ds} ', '', 1)
            msg = msg[0].upper() + msg[1:]

        elif 'creation' in log_line['message']:
            ds = log_line['message'].split()[0]
            step = 'cycle creation'
            msg = log_line['message'].replace(f'{ds} ', '', 1)
            msg = msg[0].upper() + msg[1:]

        elif 'regridding' in log_line['message']:
            ds = 'non dataset specific steps'
            step = 'regridding'
            msg = log_line['message']

        elif 'Index' in log_line['message']:
            ds = 'non dataset specific steps'
            step = 'index calculation'
            msg = log_line['message']

        if log_line['level'] == 'INFO':
            dataset_statuses[ds][step] = [('INFO', msg)]

        if log_line['level'] == 'ERROR':
            dataset_statuses[ds][step] = [('ERROR', msg)]

    # Print dataset status summaries
    for ds, steps in dataset_statuses.items():
        print(f'\033[93mPipeline status for {ds}\033[0m:')
        for _, messages in steps.items():
            for (level, message) in messages:
                if level == 'INFO':
                    print(f'\t\033[92m{message}\033[0m')
                elif level == 'ERROR':
                    print(f'\t\033[91m{message}\033[0m')

    if index_statuses:
        print('\033[93mPipeline status for index calculations\033[0m:')
        for (level, message) in index_statuses:
            if level == 'INFO':
                print(f'\t\033[92m{message}\033[0m')
            elif level == 'ERROR':
                print(f'\t\033[91m{message}\033[0m')


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
        # harv_logger = logging.getLogger(f'pipeline.{ds}.harvester')
        try:
            print(f'\033[93mRunning harvester for {ds}\033[0m')
            print(ROW)

            config_path = Path(
                f'{Path(__file__).resolve().parents[2]}/dataset_configs/{ds}/harvester_config.yaml')
            with open(config_path, "r") as stream:
                config = yaml.load(stream, yaml.Loader)

            from harvesters.harvester import harvester

            status = harvester(
                config=config, output_path=output_dir, log_time=LOG_TIME)

            log.info(f'{ds} harvesting complete. {status}')
            print('\033[92mHarvest successful\033[0m')
        except Exception as e:
            print(e)
            log.error(f'{ds} harvesting failed. {e}')

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

            config_path = Path(
                f'{Path(__file__).resolve().parents[2]}/dataset_configs/{ds}/processing_config.yaml')
            with open(config_path, "r") as stream:
                config = yaml.load(stream, yaml.Loader)

            from processors.cycle_creation import cycle_creation

            status = cycle_creation(config=config,
                                    output_path=output_dir,
                                    reprocess=reprocess,
                                    log_time=LOG_TIME)

            log.info(f'{ds} cycle creation complete. {status}')
            print('\033[92mCycle creation complete\033[0m')
        except Exception as e:
            print(e)
            log.error(f'{ds} cycle creation failed. {e}')
            print('\033[91mCycle creation failed\033[0m')
        print(ROW)


def run_cycle_regridding(src_path, output_dir, reprocess):
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

        config_path = src_path / 'processors/regridding/regridding_config.yaml'
        with open(config_path, "r") as stream:
            config = yaml.load(stream, yaml.Loader)

        from processors.regridding.regridding import regridding

        regridding(config=config, output_dir=output_dir,
                   reprocess=reprocess, log_time=LOG_TIME)

        log.info('Cycle regridding complete.')
        print('\033[92mCycle regridding complete\033[0m')
    except Exception as e:
        print(e)
        log.error(f'Cycle regridding failed. {e}')
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

    try:
        print('\033[93mRunning index calculation\033[0m')
        print(ROW)

        config_path = Path(
            src_path/'processors/indicators/indicators_config.yaml')
        with open(config_path, "r") as stream:
            config = yaml.load(stream, yaml.Loader)

        from processors.indicators.indicators import indicators

        indicators(config=config, output_path=output_dir,
                   reprocess=reprocess, log_time=LOG_TIME)

        log.info('Index calculation complete.')
        print('\033[92mIndex calculation successful\033[0m')
    except Exception as e:
        print(e)
        log.error(f'Index calculation failed: {e}')
        print('\033[91mIndex calculation failed\033[0m')
    print(ROW)


if __name__ == '__main__':

    # Make sure Solr is up and running
    verify_solr_running()

    print('\n' + ROW)
    print(' SEA LEVEL INDICATORS PIPELINE '.center(57, '='))
    print(ROW)

    # path to harvester and preprocessing folders
    pipeline_path = Path(__file__).resolve()

    PATH_TO_DATASETS = Path(f'{pipeline_path.parents[2]}/dataset_configs')
    PATH_TO_SRC = Path(pipeline_path.parents[1])
    sys.path.insert(1, str(PATH_TO_SRC))

    PARSER = create_parser()
    args = PARSER.parse_args()

    # -------------- Harvested Entry Validation --------------
    if args.harvested_entry_validation:
        harvested_entry_validation()

    # ------------------- Output directory -------------------
    if args.output_dir or not OUTPUT_DIR:
        print('\nPlease choose your output directory')

        root = tk.Tk()
        root.attributes('-topmost', True)
        root.withdraw()
        OUTPUT_DIR = Path(f'{filedialog.askdirectory()}/')

        if OUTPUT_DIR == '/':
            print('No output directory given. Exiting.')
            sys.exit()
        else:
            logs_path = Path(OUTPUT_DIR / f'logs/{LOG_TIME}')
            logs_path.mkdir(parents=True, exist_ok=True)
    else:
        if not OUTPUT_DIR.exists():
            print(f'{OUTPUT_DIR} is an invalid output directory. Exiting.')
            sys.exit()
    print(f'\nUsing output directory: {OUTPUT_DIR}')

    # ------------------ Force Reprocessing ------------------
    REPROCESS = bool(args.force_processing)

    # --------------------- Run pipeline ---------------------

    # Initialize pipeline log
    formatter = logging.Formatter(
        "{'time': '%(asctime)s', 'level': '%(levelname)s', 'message': '%(message)s'}")

    file_handler = logging.FileHandler(logs_path / 'pipeline.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    log.addHandler(file_handler)

    DATASETS = [ds.name for ds in PATH_TO_DATASETS.iterdir()
                if ds.name != '.DS_Store']

    CHOSEN_OPTION = show_menu() if args.options_menu and not REPROCESS else '1'

    # Run all
    if CHOSEN_OPTION == '1':
        for dataset in DATASETS:
            run_harvester([dataset], OUTPUT_DIR)
            run_cycle_creation([dataset], OUTPUT_DIR, REPROCESS)
        run_cycle_regridding(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)
        run_indexing(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)

    # Run harvester
    elif CHOSEN_OPTION == '2':
        for dataset in DATASETS:
            run_harvester([dataset], OUTPUT_DIR)

    # Run processing
    elif CHOSEN_OPTION == '3':
        for dataset in DATASETS:
            run_cycle_creation([dataset], OUTPUT_DIR, REPROCESS)

    # Run cycle regridding
    elif CHOSEN_OPTION == '4':
        run_cycle_regridding(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)

        # run_indexing(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)

    # Manually enter dataset and pipeline step(s)
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
            run_cycle_regridding(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)
            # run_indexing(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)
        if wanted_steps == 'all':
            run_harvester([CHOSEN_DS], OUTPUT_DIR)
            run_cycle_creation([CHOSEN_DS], OUTPUT_DIR, REPROCESS)
            run_cycle_regridding(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)
            # run_indexing(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)

    elif CHOSEN_OPTION == '6':
        run_indexing(PATH_TO_SRC, OUTPUT_DIR, REPROCESS)

    print_log(OUTPUT_DIR)
