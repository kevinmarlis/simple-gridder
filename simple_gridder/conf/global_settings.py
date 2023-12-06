import os
from pathlib import Path

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = '/Users/marlis/Developer/Measures-Cloud/alongtrack-delivery'
OUTPUT_DIR = Path(Path(ROOT_DIR).parent.parent / 'simple_grid_output/')
Path.mkdir(OUTPUT_DIR, parents=True, exist_ok=True)

os.chdir(ROOT_DIR)
