from matplotlib import pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy as cart
from glob import glob


for f in glob(f'ref_files/*pattern_and_index*'):
    pattern = f'{f.split("_")[0]}_pattern'
    ds = xr.open_dataset(f)

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