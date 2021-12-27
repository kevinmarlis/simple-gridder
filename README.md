# Sea Level Indicators

A data pipeline that calculates El Niño Southern Oscillation, Pacific Decadal Oscillation and Indian Ocean Diple sea level indicator values from along track satellite data.

## Getting Started
After cloning the repo, you should create a conda environment for the project.

```
conda env create -f SLI-pipeline/src/tools/package.yaml -n sli-pipeline
```

### Solr
You will also need a Solr instance running with a collection setup for the indicators pipeline.
```
TODO: add Solr setup steps
```
In `SLI_pipeline/utils/solr_utils.py`, define the `SOLR_HOST` and `SOLR_COLLECTION` variables based on your Solr setup.

### Initial Configuration
Copy `SLI_pipeline/configs/login.yaml.example` and fill in the login credentials. 
```
cp SLI_pipeline/configs/login.yaml.example SLI_pipeline/configs/login.yaml
```

## Running the pipeline

The pipeline defaults to running every step, but a command line argument can be passed to runt he pipeline via an options menu. This allows you to select which steps of the pipeline to perform.
```
/opt/anaconda3/envs/sli/bin/python SLI_pipeline/run_pipeline.py --options_menu
```

## Links
Indicators can be found at https://sealevel.jpl.nasa.gov