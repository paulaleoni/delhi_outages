# Delhi outages
Analyse electricity outages in delhi using. Welfare Analyses using bunching estimators based on kinks and notches in the electricity regulation in Delhi, India. Link to pollution data and nightlight data.

## Bunching estimation
- simulations using generated data (simulations.ipynb)
- estimation using real data (outage_bunching.ipynb)
- model estimation to recover parameters of cost function (outage_model_estimation.ipynb)
- welfare analysis (outage_welfare.ipynb)
- descriptive analysis of outage date, especially the reasons for it (outage_descriptives.ipynb)
- common function and classes (tools.py)

## Nightlight data
- retrieving nightlight data from VIIRS add a daily level (Nightly DNB Mosaic and Cloud, see: https://eogdata.mines.edu/products/vnl/#daily)
- summarise in a dataframe at grid level
- plot for a single day

## Pollution
### data_exploration: 
- explore pollution data; make one df based on stations and one df with grid cells as cross-section 

- input: env_data_0726.zip, Indialocationlist.csv

- output: stations_w_pol_loc.csv , stations_delhi.csv , grid_w_stat.csv

### maps:
- visualize using maps

- input: sh819zz8121.shp, stations_w_pol_loc.csv,stations_delhi.csv, grid_w_stat.csv

### pollution_data_02
- analysis with other dataset for comparison

### data_correlations
- check the correlations of different datasets

