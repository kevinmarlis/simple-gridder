# Sea Level Indicators Pipeline

## Running the pipeline
___
The pipeline first checks if Solr is running and accessible. It will quit with an
error if it is not.

If the `options_menu` command line argument is not provided, the pipeline will default
to running all steps. Otherwise, the following steps can be selected:
1. Run pipeline on all (from harvesting to text generation)
2. Harvest all datasets
3. Harvest single dataset
4. Perform gridding (gridding is performed across all datasets as datasets are combined into a single gridded product)
5. Calculate index values and generate txt output and plots
6. Generate txt output and plots
7. Post indicators to FTP

## Processing steps
___

### harvester.py

### cycle_gridding.py

### indicators.py

## Post processing steps
___

### txt_engine.py

### plotting/plot_generation.py

### upload_indicators.py