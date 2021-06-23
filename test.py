import xarray as xr
from datetime import datetime, timedelta
from matplotlib import pyplot as plt

path = '/Users/marlis/Developer/SLI/sealevel_output/JASON_3/harvested_granules/2019/JASON_3_20190829T005850Z.nc'

ds = xr.open_dataset(path)

eq_dt = datetime.strptime(ds.attrs['equator_time'], '%Y-%m-%d %H:%M:%S.%f')

adjusted_times = [eq_dt + timedelta(seconds=time)
                  for time in ds.time_rel_eq.values]

# print(ds.time_rel_eq.values)
# plt.plot(adjusted_times)
# plt.show()
ds.time_rel_eq.plot()
plt.show()
print(adjusted_times[0], adjusted_times[-1])
print(ds.attrs['first_meas_time'], ds.attrs['last_meas_time'])
