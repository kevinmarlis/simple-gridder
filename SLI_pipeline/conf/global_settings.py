import os
from pathlib import Path

ROOT_DIR = os.path.realpath(os.path.join(
    os.path.join(os.path.dirname(__file__), '..'), '..'))
OUTPUT_DIR = Path('/Users/marlis/Developer/SLI/sealevel_output/')
OUTPUT_DIR = Path('/Users/marlis/Developer/SLI/dev_output/')

SOLR_HOST = 'http://localhost:8983/solr/'
SOLR_COLLECTION = 'sli_dev'

os.chdir(ROOT_DIR)
