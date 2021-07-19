import xarray as xr
import os
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np
from tabulate import tabulate
import seaborn as sns
import requests


def solr_query(config, fq, sort=''):
    """
    Queries Solr database using the filter query passed in.

    Params:
        config (dict): the dataset specific config file
        fq (List[str]): the list of filter query arguments

    Returns:
        response.json()['response']['docs'] (List[dict]): the Solr docs that satisfy the query
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    query_params = {'q': '*:*',
                    'fq': fq,
                    'rows': 300000}

    if sort:
        query_params['sort'] = sort

    url = f'{solr_host}{solr_collection_name}/select?'
    response = requests.get(url, params=query_params)
    return response.json()['response']['docs']


def solr_update(config, update_body):
    """
    Updates Solr database with list of docs. If a doc contains an existing id field,
    Solr will update or replace that existing doc with the new doc.

    Params:
        config (dict): the dataset specific config file
        update_body (List[dict]): the list of docs to update on Solr

    Returns:
        requests.post(url, json=update_body) (Response): the Response object from the post call
    """

    solr_host = config['solr_host_local']
    solr_collection_name = config['solr_collection_name']

    url = f'{solr_host}{solr_collection_name}/update?commit=true'

    return requests.post(url, json=update_body)


config = {'solr_host_local': 'http://localhost:8983/solr/',
          'solr_collection_name': 'sealevel_datasets'}

fq = ['type_s:dataset', 'dataset_s:CRYOSAT_2']


path = Path('/Users/marlis/Developer/SLI/sealevel_output/indicator')

files = [x for x in path.rglob('indicators.nc') if x.is_file()]
files.sort()
ref_missions = [
    x for x in files if 'GSFC' in x.parent.name or 'JASON' in x.parent.name]
instruments = [x for x in files if x not in ref_missions]
instruments.sort()

sat_periods = {'ERS_1': (0, '1995-04-25'),
               'ERS_2': (0, '2002-05-14'),
               'ENVISAT_1': (0, '2012-01-01'),
               'CRYOSAT_2': ('2012-01-01', '2013-03-14'),
               'SARAL': (0, '2016-07-04'),
               'SENTINEL_3A': (0, '2018-05-08'),
               'SENTINEL_3B': (0, 0),
               'JASON_3': (0, 0)
               }


def mean_offset(ref_missions, instruments):
    combos = []
    gsfc_means = []
    jason_means = []

    gsfc_ds = xr.open_dataset(ref_missions[0])
    jason_ds = xr.open_dataset(ref_missions[1])

    instruments.append(ref_missions[1])
    sat_periods['JASON_3'] = (0, 0)

    for path in instruments:
        combo = path.parent.name.replace('_method_3', '')
        combos.append(f'{combo} - GSFC')

        (start, end) = sat_periods[combo]
        if start == 0:
            start = '1992-01-01'
        if end == 0:
            end = '2022-01-01'

        ds = xr.open_dataset(path)
        print(
            np.all(ds.time.values[1:] >= ds.time.values[:-1], axis=0))
        ds = ds.sel(time=slice(start, end))

        print(ds.time)
        exit()

        gsfc_slice_ds = gsfc_ds.sel(time=slice(start, end))

        interp_gsfc = gsfc_slice_ds['spatial_mean'].interp(time=ds.time.values)
        difference = (ds['spatial_mean'].values - interp_gsfc.values) * 100
        mean = np.nanmean(difference)
        median = np.nanmedian(difference)
        print(combo, mean.round(2), median.round(2))
        plt.cla()
        sns.kdeplot(difference)
        plt.axvline(x=median, color='red', label=f'median={median.round(2)}')
        plt.axvline(x=mean, color='black', label=f'mean={mean.round(2)}')
        plt.title(combo)
        plt.xlim(-2, 5)
        plt.xticks(np.arange(-2, 6, 2))
        plt.legend()
        plt.grid()
        plt.savefig(f'{combo}.png')
    plt.cla()
    # exit()
    # gsfc_means.append(np.nanmedian(difference * 100).round(2))

    # combos.append('JASON_3 - GSFC')
    # interp_gsfc = gsfc_ds['spatial_mean'].interp(time=jason_ds.time.values)
    # difference = jason_ds['spatial_mean'].values - interp_gsfc.values
    # gsfc_means.append(np.nanmedian(difference * 100).round(2))

    # table_dict = {'Satellite': combos, 'GSFC Means': gsfc_means}
    # table = tabulate(table_dict, headers='keys', tablefmt="simple")
    # print(table)


def plots(ref_missions, instruments):

    n = len(ref_missions)
    colors = plt.cm.twilight_shifted(np.linspace(.7, .3, n))
    i = 0
    labels = []
    for f in ref_missions:

        combo = f.parent.name.replace('_method_3', '')

        ds = xr.open_dataset(f)
        global_mean = ds['spatial_mean']
        l1, = plt.plot(global_mean.time, global_mean,
                       label=combo, alpha=.75, color=colors[i])
        labels.append(l1)
        i += 1
    print(type(labels[0]))
    first_legend = plt.legend(handles=labels)
    ax = plt.gca().add_artist(first_legend)

    n = len(instruments)
    colors = plt.cm.turbo(np.linspace(0, 1, n))
    i = 0
    labels = []
    for f in instruments:

        combo = f.parent.name.replace('_method_3', '')

        (start, end) = sat_periods[combo]
        if start == 0:
            start = '1992-01-01'
        if end == 0:
            end = '2022-01-01'

        ds = xr.open_dataset(f)
        ds = ds.sel(time=slice(start, end))

        global_mean = ds['spatial_mean']
        l1, = plt.plot(global_mean.time, global_mean,
                       label=combo, color=colors[i], alpha=1, linewidth=2)
        i += 1
        labels.append(l1)
    labels_ordered = [labels[2], labels[3], labels[1],
                      labels[0], labels[4], labels[5], labels[6]]
    plt.legend(handles=labels_ordered, loc='lower right')
    plt.grid()
    plt.show()


def ind_plots(ref_missions, instruments):

    n = len(instruments)
    colors = plt.cm.turbo(np.linspace(0, 1, n))
    i = 0
    for f in instruments:
        plt.cla()
        m = len(ref_missions)
        c = plt.cm.twilight_shifted(np.linspace(.7, .3, m))
        j = 0
        for g in ref_missions:

            combo = g.parent.name.replace('_method_3', '')

            ds = xr.open_dataset(g)
            global_mean = ds['spatial_mean']
            plt.plot(global_mean.time, global_mean,
                     label=combo, alpha=.75, color=c[j])
            j += 1

        combo = f.parent.name.replace('_method_3', '')

        (start, end) = sat_periods[combo]
        if start == 0:
            start = '1992-01-01'
        if end == 0:
            end = '2022-01-01'

        ds = xr.open_dataset(f)
        ds = ds.sel(time=slice(start, end))

        global_mean = ds['spatial_mean']
        plt.plot(global_mean.time, global_mean,
                 label=combo, color=colors[i], alpha=1, linewidth=2)
        i += 1

        td = (np.nanmax(global_mean.time.values) -
              np.nanmin(global_mean.time.values)).astype('timedelta64[D]')/10

        plt.legend()
        plt.grid()
        plt.xlim(np.nanmin(global_mean.time.values) - td,
                 np.nanmax(global_mean.time.values) + td)
        plt.gcf().autofmt_xdate()
        plt.savefig(f'time_series/{combo}.png')
        # print()
        # plt.show()


def plot_indicator(ref_missions):
    ind_path = '/Users/marlis/Developer/SLI/sealevel_output/indicator/COMBINATION_1_method_3/indicators.nc'
    indicator_ds = xr.open_dataset(ind_path)

    m = len(ref_missions)
    c = plt.cm.twilight_shifted(np.linspace(.7, .3, m))
    j = 0
    for g in ref_missions:

        combo = g.parent.name.replace('_method_3', '')

        ds = xr.open_dataset(g)
        global_mean = ds['spatial_mean']
        plt.plot(global_mean.time, global_mean,
                 label=combo, alpha=.75)
        j += 1

    global_mean = indicator_ds['spatial_mean']
    print(global_mean)
    plt.plot(global_mean.time, global_mean,
             label='Spatial mean', alpha=1)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


# mean_offset(ref_missions, instruments)
# plots(ref_missions, instruments)
# ind_plots(ref_missions, instruments)
plot_indicator(ref_missions)
