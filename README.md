# Sea Level Indicators

A data pipeline that calculates El Niño Southern Oscillation, Pacific Decadal Oscillation and Indian Ocean Diple sea level indicator values from along track satellite data.

## Getting Started
After cloning the repo, you should create a conda environment for the project.

```
conda env create -f SLI_pipeline/conf/environment.yaml -n sli-pipeline
```


### Initial Configuration
Copy `SLI_pipeline/conf/login.yaml.example` and fill in the login credentials. 
```
cp SLI_pipeline/conf/login.yaml.example SLI_pipeline/conf/login.yaml
```

Make sure ref_files contains the required files. See `ref_files/README.md` for details.

Define an output directory in `SLI_pipeline/conf/global_settings.py`, if not using standard output directory. 


## Running the pipeline

The pipeline defaults to running every step, but a command line argument can be passed to runt he pipeline via an options menu. This allows you to select which steps of the pipeline to perform.
```
/opt/anaconda3/envs/sli-pipeline/bin/python SLI_pipeline/run_pipeline.py --options_menu
```

## Links
Indicators can be found at https://sealevel.jpl.nasa.gov