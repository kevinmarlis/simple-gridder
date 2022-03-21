from matplotlib import pyplot as plt
import os
import xarray as xr
import cartopy.crs as ccrs
import cartopy as cart

ref_path = '/Users/marlis/Developer/SLI/Sea-Level-Indicators/ref_files'
ben_files = [f for f in os.listdir(ref_path) if 'pattern_and_index' in f]

for f in ben_files:
    pattern = f'{f.split("_")[0]}_pattern'
    ds = xr.open_dataset(f'{ref_path}/{f}')

    lons = ds.Longitude.values
    lats = ds.Latitude.values
    data = ds[pattern].values

    fig = plt.figure(figsize=(12, 6), dpi=90)

    ax = fig.add_subplot(
        1, 1, 1, projection=ccrs.Robinson(central_longitude=180))

    ax.set_global()

    # ax.add_feature(cart.feature.LAND)

    im = plt.contourf(lons, lats, data, 60,
                      transform=ccrs.PlateCarree())
    ax.stock_img()
    ax.coastlines()
    ax.add_feature(cart.feature.OCEAN)

    plt.colorbar(im, ax=ax)
    plt.show()
