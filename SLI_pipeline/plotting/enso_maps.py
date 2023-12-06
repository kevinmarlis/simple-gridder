import xarray as xr
from matplotlib import pyplot as plt
from matplotlib import colors
import matplotlib.ticker as mticker

from datetime import datetime
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import warnings
warnings.filterwarnings('ignore')


def make_akiko_cmap() -> colors.ListedColormap:
    '''
    Converts colorscale txt file to mpl 
    '''
    values = []
    with open('/Users/marlis/Developer/SLI/Sea-Level-Indicators/ref_files/akiko_colorscale.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            vals = line.split()
            row = [float(v)/256 for v in vals]
            row.append(1)
            values.append(row)
    return colors.ListedColormap(values, name='akiko_cmap')

akiko_cmap = make_akiko_cmap()

enso_grid_paths = glob(f'/Users/marlis/Developer/SLI/sli_output/ENSO_grids/*.nc')
enso_grid_paths.sort()

outdir = '/Users/marlis/Developer/SLI/sli_output/ENSO_maps'

def date_sat_map(date: datetime.date) -> str:
    '''
    TOPEX/Poseidon -> Jason-1:  			14 May 2002
    Jason-1 -> Jason-2:						12 Jul 2008
    Jason-2 -> Jason-3:						18 Mar 2016
    Jason-3 -> Sentinel-6 Michael Freilich:	07 Apr 2022
    '''
    topex = (date(1992,1,1), date(2002,5,14))
    j1 = (date(2002,5,14), date(2008,7,12))
    j2 = (date(2008,7,12), date(2016,3,18))
    j3 = (date(2016,3,18), date(2022,4,7))
    s6 = (date(2022,4,7), date.today())
    
    if date >= topex[0] and date < topex[1]:
        return 'TOPEX/Poseidon'
    if date >= j1[0] and date < j1[1]:
        return 'Jason-1'
    if date >= j2[0] and date < j2[1]:
        return 'Jason-2'
    if date >= j3[0] and date < j3[1]:
        return 'Jason-3'
    if date >= s6[0] and date < s6[1]:
        return 'Sentinel-6 Michael Freilich'

def plot_orth(enso_ds, date, satellite, outdir, vmin=-180, vmax=180):
    date_str = datetime.strftime(date, '%b %d %Y').upper()
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-150, 10))
    
    ax.pcolormesh(enso_ds.longitude, enso_ds.latitude, enso_ds.SSHA, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=akiko_cmap, shading='nearest')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
    ax.add_feature(cfeature.LAND, facecolor='dimgrey', zorder=10)
    ax.coastlines(zorder=11)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.75,zorder=12)
    
    gl.xlocator = mticker.FixedLocator([])
    gl.ylocator = mticker.FixedLocator([0])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    fig.set_facecolor('black')
    
    fig.text(-.1, 1.02, date_str, color='white', ha='left', va='top', size=20, transform=ax.transAxes)
    if satellite == 'Sentinel-6 Michael Freilich':
        fig.text(1.1, 1.02, satellite.split(' ')[0], color='white', ha='right', va='top', 
                 size=20, transform=ax.transAxes, wrap=True)
        fig.text(1.1, 0.98, satellite.split('Sentinel-6 ')[-1], color='white', ha='right', va='top', 
                 size=20, transform=ax.transAxes, wrap=True)
    else:
        fig.text(1.1, 1.02, satellite, color='white', ha='right', va='top', size=20, 
                 transform=ax.transAxes, wrap=True)

    outpath = f'{outdir}/ENSO_ortho/ENSO_ortho_{str(date).replace("-","")}.png'
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.5)



def plot_plate(enso_ds, date, satellite, outdir, vmin=-180, vmax=180):
    date_str = datetime.strftime(date, '%b %d %Y').upper()

    fig = plt.figure(figsize=(20,8))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree(-180))
    
    g = plt.pcolormesh(enso_ds.longitude, enso_ds.latitude, enso_ds.SSHA, transform=ccrs.PlateCarree(), 
                       vmin=vmin, vmax=vmax, cmap=akiko_cmap)
    
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
    ax.add_feature(cfeature.LAND, facecolor='dimgrey', zorder=10)
    ax.coastlines(zorder=11)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='gray', alpha=.5, linestyle='--', zorder=15)
    
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlocator = mticker.FixedLocator([40, 80, 120, 160, -160, -120, -80, -40])
    ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
    ax.xaxis.set_minor_formatter(LONGITUDE_FORMATTER)
    gl.xlabel_style = {'size': 14}
    gl.ylabel_style = {'size': 14}

    plt.title(f'{satellite} Sea Level Residuals {date_str}', size=16)
    cb = plt.colorbar(g, orientation="horizontal", shrink=0.5, aspect=30, pad=0.1)
    cb.set_label('MM', fontsize=14)
    cb.ax.tick_params(labelsize=12) 
    fig.tight_layout()

    outpath = f'{outdir}/ENSO_plate/ENSO_plate_{str(date).replace("-","")}.png'
    plt.savefig(outpath, bbox_inches='tight', pad_inches=0.5)


def plot_orth_enso(enso_ds, date, outdir, vmin=-180, vmax=180):
    date_str = datetime.strftime(date, '%d %b %Y')
    fig = plt.figure(figsize=(14,10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Orthographic(-150, 10))
    ax.pcolormesh(enso_ds.longitude, enso_ds.latitude, enso_ds.SSHA, transform=ccrs.PlateCarree(), vmin=vmin, vmax=vmax, cmap=my_cmap, shading='nearest')
    ax.add_feature(cfeature.OCEAN, facecolor='lightgrey')
    ax.add_feature(cfeature.LAND, facecolor='dimgrey', zorder=10)
    ax.coastlines(zorder=11)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=2, color='black', alpha=0.75,zorder=12)
    gl.xlocator = mticker.FixedLocator([])
    gl.ylocator = mticker.FixedLocator([0])
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    fig.set_facecolor('black')
    fig.text(.6375, 1, date_str, color='white', ha='right', va='bottom', transform=ax.transAxes, fontname='Arial', fontsize=52)
    
    ax.set_ylim(-3000000,2000000)
    fig.tight_layout()

    outpath = f'{outdir}/ENSO_ortho/ENSO_ortho_{str(date).replace("-","")}.png'
    plt.savefig(outpath, bbox_inches='tight', pad_inches=.75)

def plot_animation_frames():
    '''
    REQUIRES sli-pipeline ENV
    '''
    for f in enso_grid_paths:
        ds = xr.open_dataset(f)
        date_dt = datetime.strptime(str(ds.time.values)[:10], '%Y-%m-%d').date()
        logging.info(date_dt)
        plot_orth_enso(ds, date_dt, outdir, -130, 130)
        
def plot_website_images():
    '''
    REQUIRES simple-gridder ENV
    '''
    for f in enso_grid_paths:
        ds = xr.open_dataset(f)
        date_dt = datetime.strptime(str(ds.time.values)[:10], '%Y-%m-%d').date()
        print(date_dt)
        satellite = date_sat_map(date_dt)
        
        plot_orth(ds, date_dt, satellite, outdir)
        plot_plate(ds, date_dt, satellite, outdir)
