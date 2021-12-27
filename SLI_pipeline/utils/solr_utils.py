import os
import requests
from datetime import datetime

SOLR_HOST = 'http://localhost:8983/solr/'
SOLR_COLLECTION = 'sli_dev'


def solr_query(fq, sort=''):
    query_params = {'q': '*:*',
                    'fq': fq,
                    'rows': 300000}
    if sort:
        query_params['sort'] = sort

    url = f'{SOLR_HOST}{SOLR_COLLECTION}/select?'
    response = requests.get(url, params=query_params)
    return response.json()['response']['docs']


def solr_query_boost(fq, boost_function):
    query_params = {'q': '*:*',
                    'fq': fq,
                    'bf': boost_function,
                    'defType': 'edismax',
                    'rows': 300000,
                    'sort': 'date_s asc'}

    url = f'{SOLR_HOST}{SOLR_COLLECTION}/select?'
    response = requests.get(url, params=query_params)
    return response.json()['response']['docs']


def solr_update(update_body, r=False):
    url = f'{SOLR_HOST}{SOLR_COLLECTION}/update?commit=true'
    response = requests.post(url, json=update_body)
    if r:
        return response


def ping_solr():
    requests.get(f'{SOLR_HOST}{SOLR_COLLECTION}/admin/ping')
    return
