from datetime import datetime

import numpy as np
import xarray as xr

HEADERS = 'HDR Sea Surface Height Anomaly Indicator Data\n\
HDR\n\
HDR This file contains Sea Surface Height Anomaly (SSHA) indicator data computed at \n\
HDR the NASA Jet Propulsion Laboratory (JPL) for El Niño-Southern Oscillation (ENSO), \n\
HDR Pacific Decadal Oscillation (PDO), and Indian Ocean Dipole (IOD). The indicator \n\
HDR values were calculated using a combination of  MEaSUREs Gridded Sea Surface Height\n\
HDR Anomalies Version 1812 data (https://podaac.jpl.nasa.gov/dataset/SEA_SURFACE_HEIGHT_ALT_GRIDS_L4_2SATS_5DAY_6THDEG_V_JPL1812)\n\
HDR and processed along-track Sea Surface Height Anomalies from Jason-3 and \n\
HDR SENTINEL-3B downloaded from the Radar Altimeter Database System \n\
HDR (RADS: http://rads.tudelft.nl/rads/rads.shtml). All data is placed onto the same\n\
HDR 0.5-degree latitude longitude grid, creating a consistent data record regardless\n\
HDR of the source.\n\
HDR\n\
HDR Indicator values were calculated using cyclostationary empirical orthogonal functions (CSEOFs; Kim et al., 2015)\n\
HDR computed by decomposing the gridded sea surface height anomalies over the time period from 1993 to 2019.\n\
HDR After removing the linear trend from each individual gridded location, three sets of regional CSEOFs were\n\
HDR generated, one each for ENSO, PDO and IOD. In each case, the dominant statistical mode represents the seasonal\n\
HDR cycle. The second most dominant mode represents the variability explained by each respective indicator and is\n\
HDR referred to as the “indicator mode”. The seasonal mode and indicator mode are then projected onto the along-track\n\
HDR sea surface height anomalies to produce the indicator time series through the most current date.\n\
HDR\n\
HDR For more information on how the data were generated please refer to:\n\
HDR K.Y. Kim, B. Hamlington, H. Na, “Theoretical foundation of cyclostationary EOF analysis\n\
HDR for geophysical and climatic variables: concepts and examples”, 2015. Earth-science reviews, 150, 201-218.\n\
HDR\n\
HDR P. Kumar, B. Hamlington, S. Cheon, W. Han, and P. Thompson, “20th Century Multivariate \n\
HDR Indian Ocean Regional Sea Level Reconstruction,” J. Geophys. Res. Oceans, vol. 125,\n\
HDR no. 10, Oct. 2020, doi: 10.1029/2020jc016270.\n\
HDR\n\
HDR B. D. Hamlington et al., “The Dominant Global Modes of Recent Internal Sea Level\n\
HDR Variability,” J. Geophys. Res. Oceans, vol. 124, no. 4, pp. 2750–2768, Apr. 2019,\n\
HDR doi: 10.1029/2018jc014635.\n\
HDR\n\
HDR ====================\n\
HDR\n\
HDR Sea Surface Height Anomaly Indicators\n\
HDR\n\
HDR column description\n\
HDR 1 year+fraction of year\n\
HDR 2 ENSO (El Nino Southern Oscillation) indicator values\n\
HDR 3 PDO (Pacific Decadal Oscillation) indicator values\n\
HDR 4 IOD (Indian Ocean Dipole) indicator values\n\
HDR\n\
HDR Missing or bad value flag: 9.96921E36f\n\
HDR\n\
HDR Header_End-------------------------------------\n'


def dt_to_dec(dt):
    '''
    Transforms datetime values to year decimal values.
    '''

    datetime_dt = datetime.strptime(
        np.datetime_as_string(dt, unit='s'), '%Y-%m-%dT%H:%M:%S')

    year_start = datetime(datetime_dt.year, 1, 1)
    year_end = year_start.replace(year=datetime_dt.year + 1)

    return datetime_dt.year + ((datetime_dt - year_start).total_seconds() /
                               float((year_end - year_start).total_seconds()))


def create_lines(dates, ds):
    '''
    Creates list of formatted strings consisting of
    date, enso, pdo, and iod values per string.
    '''
    lines = []

    zipped_vals = zip(dates, ds['enso_index'].values,
                      ds['pdo_index'].values, ds['iod_index'].values)

    for (date, enso, pdo, iod) in zipped_vals:
        data_line = '{0:<12.7f} {1:>12f} {2:>12f} {3:>12f}\n'.format(
            date, enso, pdo, iod)
        lines.append(data_line)

    return lines


def generate_txt(output_path, ind_path):

    ds = xr.open_dataset(ind_path)

    # Get times in decimal format
    dates = []
    for date in ds.time.values:
        dates.append(dt_to_dec(date))

    # Translate dates and indicator values into individual text lines
    lines = create_lines(dates, ds)

    with open(output_path, 'w') as f:
        f.write(HEADERS)

        for line in lines:
            f.write(line)


def main(output_dir):
    print('Generating txt file from indicators')

    ind_path = output_dir / 'indicator/indicators.nc'

    output_path = output_dir / f'indicator/indicator_data.txt'

    generate_txt(output_path, ind_path)
