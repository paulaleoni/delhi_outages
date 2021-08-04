'''
map of India with stations

source: https://geodata.lib.utexas.edu/catalog/stanford-sh819zz8121
'''

from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely import wkt

wd=Path.cwd()

# load data
map = gpd.read_file(wd.parent/'data'/'shp'/'sh819zz8121.shp')
stations = pd.read_csv(wd.parent/'data'/'data_transformed'/'stations_w_pol_loc.csv')
stations_delhi = pd.read_csv(wd.parent/'data'/'data_transformed'/'stations_delhi.csv')
grid = pd.read_csv(wd.parent/'data'/'data_transformed'/'grid_w_stat.csv')

delhi = {'city':'delhi', 'geometry':Point(77.23149, 28.65195)}
# source: https://www.geodatos.net/en/coordinates/india/delhi
delhi = gpd.GeoDataFrame(delhi, index=[0], crs='EPSG:4326')


#*#########################
#! DATA PREP
#*#########################
def prep_data(df, col):
    '''
    CSVs created contain an Unnamed: 0 column and columns containing geometries are imported as strings. This function drop the former and transforms the latter.
    
    *df=df to be transformed
    *col=list of column(s) containing geometries
    '''
    if 'Unnamed: 0' in df.columns: 
        df=df.drop('Unnamed: 0', axis=1)    
    for c in col: 
        df[c]=df[c].apply(wkt.loads)
    
    return df

stations = prep_data(stations, ['geometry'])
stations_delhi = prep_data(stations_delhi, ['geometry', 'grid_geometry'])
grid = prep_data(grid, ['geometry'])

# make a geo df
stations = gpd.GeoDataFrame(stations)
stations_delhi = gpd.GeoDataFrame(stations_delhi)
grid = gpd.GeoDataFrame(grid)


#some stations are not in India, drop those
stations_clear = stations
for row in stations_clear.index:
    if stations_clear.loc[row,'geometry'].x < 50:
        stations_clear = stations_clear.drop(row) 
    elif stations_clear.loc[row,'geometry'].x > 100:
        stations_clear = stations_clear.drop(row)      
stations_clear = stations_clear.reset_index(drop=True)

# 202 dropped
len(stations) - len(stations_clear) 

# extract unclear points
x = [x for x in stations.station.tolist() if x not in stations_clear.station.tolist()]
test = stations[stations.apply(lambda row: row.station in x, axis =1)]
test.stateName.unique()
test.countryName.unique()

# extract polygon of delhi in map
point_delhi = delhi['geometry'][0]
in_delhi = map.apply(lambda row: point_delhi.within(row.geometry), axis=1)
map_delhi = map[in_delhi].reset_index(drop=True)


#*#########################
#! PLOTS
#*#########################

# all stations
fig, ax = plt.subplots(figsize=(12,12))
map.plot(color='white', edgecolor='grey', ax=ax)
stations.plot(color='darkred',markersize=8, ax=ax, label='stations')
plt.title('India with all stations')
plt.text(108,80,f'{len(stations)} stations')
plt.axis('off')
plt.legend()
plt.show()
fig.savefig('visualizations/India_stat_all.png')

# stations in India
fig, ax = plt.subplots(figsize=(12,12))
map.plot(color='white', edgecolor='grey', ax=ax)
stations_clear.plot(color='darkred',markersize=8, ax=ax, label='stations')
delhi.plot(color='green', ax=ax, label='delhi')
plt.title('India')
plt.text(94.5,36,f'{len(stations_clear)} stations')
plt.axis('off')
plt.legend()
plt.show()
fig.savefig('visualizations/India_stat.png')

# stations in delhi
fig, ax = plt.subplots(figsize=(12,12))
map_delhi.plot(color='white', edgecolor='grey', ax=ax)
stations_delhi.plot(color='darkred',markersize=10, ax=ax, label='stations')
plt.title('Delhi')
plt.text(77.31,28.85,f'{len(stations_delhi)} stations')
plt.axis('off')
plt.legend()
plt.show()
fig.savefig('visualizations/Delhi_stat.png')

## add grid
fig, ax = plt.subplots(figsize=(12,12))
c = grid['PM25_value']
map_delhi.plot(color='white', edgecolor='black', ax=ax)
grid.plot(edgecolor='grey', color='white', ax=ax, label='grid', cmap = 'RdPu')
cbar = plt.colorbar()
stations_delhi.plot(color='darkred',markersize=10, ax=ax, label='stations')
plt.axis('off')
plt.legend()
plt.show()
fig.savefig('visualizations/Delhi_stat_grid.png')