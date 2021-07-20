# Dataset Configs

Directory containing config files for harvesting and processing datasets. Subdirectories exist per dataset and should follow this convention:

### |-- DATASET_NAME
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- harvester_config.yaml
### &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|-- processing_config.yaml

&nbsp;
___

## harvester_config.yaml
These config files contain start/end date values, the name to be used for the dataset (should be the same as the DATASET_NAME directory above), metadata regarding the dataset itself, and the location of the SOLR database.

The harvester config file also dictates where the dataset will be harvested from. Most datasets are available on a restricted PODAAC drive. The harvester type *PODAAC Drive* works for these datasets. The gridded 1812 dataset is publicly available using PODAAC's granule search. The *podaac* harvester type works for datasets available there. Follow the 1812 example config using the new dataset's PODAAC ID. Lastly, the GSFC reference mission data is hosted locally and uses the *local* harvester.


## processing_config.yaml
These config files contain the same dataset name as used above, the length in days for a cycle (typically 30 or 10 - whichever ensures global coverage), and SOLR database information. 