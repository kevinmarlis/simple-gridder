from pathlib import Path
import logging
import logging.config
import warnings
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from utils import solr_utils

logging.config.fileConfig(f'logs/log.ini',
                          disable_existing_loggers=False)
log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def generate_plots(output_dir, ind_path):
    vars = ['enso_index', 'pdo_index', 'iod_index', 'spatial_mean']
    ds = xr.open_dataset(ind_path)
    end_time = ds.time.values[-1]
    start_time = end_time - np.timedelta64(365*5, 'D')
    pdo_start_time = end_time - np.timedelta64(365*10, 'D')
    spatial_start_time = end_time - np.timedelta64(365*7, 'D')

    # Query solr to get previous indicator time
    fq = ['type_s:indicator']
    r = solr_utils.solr_query(fq)
    if 'prior_end_dt' in r[0].keys():
        slice_start = np.datetime64(r[0]['prior_end_dt'])
    else:
        slice_start = None

    output_path = output_dir / 'indicator/plots'
    output_path.mkdir(parents=True, exist_ok=True)

    for var in vars:

        if 'pdo' in var:
            var_ds = ds[var].sel(time=slice(pdo_start_time, end_time))
            delta = np.timedelta64(120, 'D')
        elif 'spatial' in var:
            var_ds = ds[var].sel(time=slice(spatial_start_time, end_time))
            delta = np.timedelta64(90, 'D')
        else:
            var_ds = ds[var].sel(time=slice(start_time, end_time))
            delta = np.timedelta64(60, 'D')

        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(10, 5))

        if 'spatial_mean' not in var:
            plt.hlines(
                y=0, xmin=var_ds.time[0]-delta, xmax=var_ds.time[-1]+delta, color='black', linestyle='-')
            max_val = max(var_ds.values)
            plt.ylim(0-max_val-.25, max_val+.25)
            plt.xlim(var_ds.time[0]-delta, var_ds.time[-1]+delta)
        else:
            var_ds = var_ds * 100
            plt.ylabel('cm')

        plt.plot(var_ds.time, var_ds, label='Indicator', linewidth=3)

        if slice_start:
            var_slice_ds = ds[var].sel(time=slice(slice_start, end_time))

            if 'spatial_mean' in var:
                var_slice_ds = var_slice_ds * 100

            plt.plot(var_slice_ds.time, var_slice_ds,
                     label='New Indicator Data', linewidth=3)

        plt.grid()
        plt.title(var)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f'{output_path}/{var}.png', dpi=150)
        # plt.show()
        plt.cla()


def main(output_dir):
    print('\nGenerating plots from indicators')

    ind_path = output_dir / 'indicator/indicators.nc'

    generate_plots(output_dir, ind_path)
