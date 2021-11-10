from pathlib import Path
import numpy as np
from numpy.core.fromnumeric import size
import xarray as xr
from matplotlib import pyplot as plt


def generate_plots(output_dir, ind_path):
    vars = ['enso_index', 'pdo_index', 'iod_index', 'spatial_mean']
    vars = ['spatial_mean']
    ds = xr.open_dataset(ind_path)
    end_time = ds.time.values[-1]
    start_time = end_time - np.timedelta64(365*5, 'D')
    pdo_start_time = end_time - np.timedelta64(365*10, 'D')
    spatial_start_time = end_time - np.timedelta64(365*7, 'D')

    for var in vars:
        output_path = output_dir / 'indicator/plots'
        output_path.mkdir(parents=True, exist_ok=True)

        if 'pdo' in var:
            var_ds = ds[var].sel(time=slice(pdo_start_time, end_time))
            delta = np.timedelta64(120, 'D')
        elif 'spatial' in var:
            var_ds = ds[var].sel(time=slice(spatial_start_time, end_time))
            delta = np.timedelta64(90, 'D')
        else:
            var_ds = ds[var].sel(time=slice(start_time, end_time))
            delta = np.timedelta64(60, 'D')

        slice_start = np.datetime64('2021-08-18T00:00:00.000000000')
        var_slice_ds = ds[var].sel(time=slice(slice_start, end_time))

        if 'spatial_mean' in var:
            var_ds = var_ds * 100
            var_slice_ds = var_slice_ds * 100

        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(10, 5))

        if 'spatial_mean' not in var:
            plt.hlines(
                y=0, xmin=var_ds.time[0]-delta, xmax=var_ds.time[-1]+delta, color='black', linestyle='-')
            plt.plot(var_ds.time, var_ds, label='Indicator', linewidth=3)
            plt.plot(var_slice_ds.time, var_slice_ds,
                     label='New Indicator Data', linewidth=3)
            max_val = max(max(var_ds.values), max(var_slice_ds.values))
            plt.ylim(0-max_val-.25, max_val+.25)
            plt.xlim(var_ds.time[0]-delta, var_ds.time[-1]+delta)
        else:
            plt.plot(var_ds.time, var_ds, label='Indicator', linewidth=3)
            plt.plot(var_slice_ds.time, var_slice_ds,
                     label='New Indicator Data', linewidth=3)
            plt.ylabel('cm')
        plt.grid()
        plt.title(var)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.tight_layout()
        plt.savefig(f'{var}.png', dpi=150)
        # plt.show()
        plt.cla()


def main(output_dir):
    print('Generating plots from indicators')

    ind_path = output_dir / 'indicator/indicators.nc'
    # DELETE ME BELOW
    ind_path = Path('/Users/marlis/Developer/SLI/indicators/indicators.nc')

    generate_plots(output_dir, ind_path)


if __name__ == '__main__':
    main(Path('/Users/marlis/Developer/SLI/sealevel_output'))
