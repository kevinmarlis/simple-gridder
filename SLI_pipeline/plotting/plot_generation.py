from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt


def generate_plots(output_dir, ind_path):
    vars = ['enso_index', 'pdo_index', 'iod_index']
    ds = xr.open_dataset(ind_path)
    end_time = ds.time.values[-1]
    start_time = end_time - np.timedelta64(365*2, 'D')

    for var in vars:
        output_path = output_dir / 'indicator/plots'
        output_path.mkdir(parents=True, exist_ok=True)

        var_ds = ds[var].sel(time=slice(start_time, end_time))

        plt.plot(var_ds.time, var_ds, label='Indicator')
        plt.grid()
        plt.title(var)
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.savefig(f'{output_path}/{var}.png')
        plt.cla()


def main(output_dir):
    print('Generating plots from indicators')

    ind_path = output_dir / 'indicator/indicators.nc'

    generate_plots(output_dir, ind_path)


if __name__ == '__main__':
    main(Path('/Users/marlis/Developer/SLI/sealevel_output'))
