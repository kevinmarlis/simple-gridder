"""
"""

import logging
from argparse import ArgumentParser
from pathlib import Path

import yaml

import txt_engine
import upload_indicators
from conf.global_settings import OUTPUT_DIR
from cycle_gridding import cycle_gridding
from harvester import harvester
from indicators import indicators
from logs.logconfig import configure_logging
from plotting import plot_generation

configure_logging(file_timestamp=False)

if not Path.is_dir(OUTPUT_DIR):
    logging.fatal('Output directory does not exist. Exiting.')
    exit()
logging.debug(f'\nUsing output directory: {OUTPUT_DIR}')

def create_parser():
    """
    Creates command line argument parser

    Returns:
        parser (ArgumentParser): the ArgumentParser object
    """
    parser = ArgumentParser()

    parser.add_argument('--options_menu', default=False, action='store_true',
                        help='Display option menu to select which steps in the pipeline to run.')

    # parser.add_argument('-h', '--harvest', type=str, default='', dest='harvest_dataset',
    #                 help='Dataset to harvest. If no dataset given, will harvest all.')
    # parser.add_argument('-gc', '--grid_cycles', type=str, default='', dest='grid_cycles',
    #                 help='Dataset to harvest')

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


def run_harvester(datasets, configs, output_dir):
    """
        Calls the harvester with the dataset specific config file path for each
        dataset in datasets.

        Parameters:
            datasets (List[str]): A list of dataset names.
            output_dir (Path): The path to the output directory.
    """
    for ds in datasets:
        try:
            ds_config = configs[ds]
            status = harvester(ds_config, output_dir)
            logging.info(f'{ds} harvesting complete. {status}')
        except Exception as e:
            logging.exception(f'{ds} harvesting failed. {e}')


def run_cycle_gridding(output_dir):
    try:
        cycle_gridding(output_dir)
        logging.info('Cycle gridding complete.')
    except Exception as e:
        logging.exception(f'Cycle gridding failed. {e}')


def run_indexing(output_dir) -> bool:
    success = False
    try:
        success = indicators(output_dir)
        logging.info('Index calculation complete.')
    except Exception as e:
        logging.error(f'Index calculation failed: {e}')

    return success


def run_post_processing(output_dir):
    try:
        plot_generation.main(output_dir)
    except Exception as e:
        logging.error(f'Plot generation failed: {e}')

    try:
        txt_engine.main(OUTPUT_DIR)
        logging.info('Index txt file creation complete.')
    except Exception as e:
        logging.error(f'Index txt file creation failed: {e}')


def post_to_ftp(output_dir):
    try:
        upload_indicators.main(output_dir)
        logging.info('Index txt file upload complete.')
    except Exception as e:
        logging.error(f'Index txt file upload failed: {e}')


if __name__ == '__main__':

    print(' SEA LEVEL INDICATORS PIPELINE '.center(57, '='))

    PARSER = create_parser()
    args = PARSER.parse_args()

    # --------------------- Run pipeline ---------------------

    with open(Path(f'conf/datasets.yaml'), "r") as stream:
        config = yaml.load(stream, yaml.Loader)
    configs = {c['ds_name']: c for c in config}

    DATASET_NAMES = list(configs.keys())

    CHOSEN_OPTION = show_menu() if args.options_menu else '1'

    # Run harvesting, gridding, indexing, post processing
    if CHOSEN_OPTION == '1':
        for dataset in DATASET_NAMES:
            run_harvester([dataset], configs, OUTPUT_DIR)
        run_cycle_gridding(OUTPUT_DIR)
        if run_indexing(OUTPUT_DIR):
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
        logging.info(f'\nHarvesting {CHOSEN_DS} dataset')

        run_harvester([CHOSEN_DS], configs, OUTPUT_DIR)

    # Run gridding
    elif CHOSEN_OPTION == '4':
        run_cycle_gridding(OUTPUT_DIR)

    # Run indexing (and post processing)
    elif CHOSEN_OPTION == '5':
        if run_indexing(OUTPUT_DIR):
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
