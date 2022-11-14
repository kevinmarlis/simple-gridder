import os
from pathlib import Path

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = Path(Path(ROOT_DIR).parent.parent / 'sli_output/')
Path.mkdir(OUTPUT_DIR, parents=True, exist_ok=True)

FILE_FORMAT = '.h5'

os.chdir(ROOT_DIR)
