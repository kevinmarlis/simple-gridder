# Dataset Configs

Directory containing config files per dataset. 
___

## harvester_config.yaml
These config files contain start/end date values, the name to be used for the dataset (should be the same as the DATASET_NAME directory above), metadata regarding the dataset itself, and the location of the SOLR database.

The harvester config file also dictates where the dataset will be harvested from. Along track instrument datasets are available on a restricted PODAAC drive. The harvester type *PODAAC Drive* works for these datasets. The gridded 1812 dataset is publicly available using PODAAC's granule search. The *podaac* harvester type works for datasets available there. Follow the 1812 example config using the new dataset's PODAAC ID. Lastly, the *local* harvester works for local data, but will require some retooling depending on the structure of the local data.


## processing_config.yaml
These config files contain the same dataset name as used above, the length in days for a cycle (typically 30 or 10 - whichever ensures global coverage), and SOLR database information. 