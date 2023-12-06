import logging
from argparse import ArgumentParser
from pathlib import Path

import yaml

from txt_engine import generate_txt
from conf.global_settings import OUTPUT_DIR
from cycle_gridding import cycle_gridding
from indicators import indicators
from logs.logconfig import configure_logging
from plotting.plot_generation import generate_plots
import enso_grids
import monthly_enso_grids


def create_parser():
    """
    Creates command line argument parser

    Returns:
        parser (ArgumentParser): the ArgumentParser object
    """
    parser = ArgumentParser()

    parser.add_argument('--options_menu', default=False, action='store_true',
                        help='Display option menu to select which steps in the pipeline to run.')
    
    parser.add_argument('--log_level',  default='INFO', help='sets the log level')

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
        print('2) Perform gridding')
        print('3) Calculate index values and generate txt output and plots')
        print('4) Generate txt output and plots')
        print('5) Generate ENSO grids and monthly mean ENSO grids')
        
        selection = input('Enter option number: ')

        if selection in ['1', '2', '3', '4', '5']:
            return selection
        print(f'Unknown option entered, "{selection}", please enter a valid option\n')


def run_cycle_gridding():
    try:
        cycle_gridding()
        logging.info('Cycle gridding complete.')
    except Exception as e:
        logging.exception(f'Cycle gridding failed. {e}')


def run_indexing() -> bool:
    success = False
    try:
        success = indicators()
        logging.info('Index calculation complete.')
    except Exception as e:
        logging.error(f'Index calculation failed: {e}')
    return success


def run_post_processing():
    try:
        generate_plots()
    except Exception as e:
        logging.error(f'Plot generation failed: {e}')

    try:
        generate_txt()
        logging.info('Index txt file creation complete.')
    except Exception as e:
        logging.error(f'Index txt file creation failed: {e}')
        
def run_enso_grids():
    try:
        enso_grids()
    except Exception as e:
        logging.error(f'Enso gridding failed: {e}')
    try:
        monthly_enso_grids()
    except Exception as e:
        logging.error(f'Enso gridding failed: {e}')


if __name__ == '__main__':

    print(' SEA LEVEL INDICATORS PIPELINE '.center(57, '='))

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

    DATASET_NAMES = list(configs.keys())

    CHOSEN_OPTION = show_menu() if args.options_menu else '1'

    # Run harvesting, gridding, indexing, post processing
    if CHOSEN_OPTION == '1':
        run_cycle_gridding()
        if run_indexing():
            run_post_processing()

    # Run gridding
    elif CHOSEN_OPTION == '2':
        run_cycle_gridding()

    # Run indexing (and post processing)
    elif CHOSEN_OPTION == '3':
        if run_indexing():
            run_post_processing()

    # Run post processing
    elif CHOSEN_OPTION == '4':
        run_post_processing()
        
    elif CHOSEN_OPTION == '5':
        run_enso_grids()
