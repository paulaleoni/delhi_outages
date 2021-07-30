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

# prepare data
stations = stations.drop('Unnamed: 0', axis=1)   
stations['geometry'] = stations.apply(lambda row: Point(row.lon, row.lat), axis=1)

#some stations are not in India, drop those
stations = gpd.GeoDataFrame(stations)
for row in stations.index:
    if stations.loc[row,'geometry'].x < 50:
        stations = stations.drop(row) 
    elif stations.loc[row,'geometry'].x > 100:
        stations = stations.drop(row)      
stations = stations.reset_index(drop=True)

# add delhi
delhi = {'city':'delhi', 'geometry':Point(77.23149, 28.65195)}
# source: https://www.geodatos.net/en/coordinates/india/delhi
delhi = gpd.GeoDataFrame(delhi, index=[0], crs='EPSG:4326')


fig, ax = plt.subplots(figsize=(12,12))
map.plot(color='white', edgecolor='grey', ax=ax)
stations.plot(color='darkred',markersize=8, ax=ax, label='stations')
delhi.plot(color='green', ax=ax, label='delhi')
plt.axis('off')
plt.legend()
plt.show()

# focus on delhi

# keep only polygon of delhi
point_delhi = delhi['geometry'][0]
in_delhi = map.apply(lambda row: point_delhi.within(row.geometry), axis=1)
map_delhi = map[in_delhi].reset_index(drop=True)

# stations inside polygon
stations_delhi = stations.apply(lambda row: row.geometry.within(map_delhi.geometry[0]), axis=1)
stations_delhi = stations[stations_delhi]
# 171 stations

fig, ax = plt.subplots(figsize=(12,12))
map_delhi.plot(color='white', edgecolor='grey', ax=ax)
stations_delhi.plot(color='darkred',markersize=10, ax=ax, label='stations')
plt.axis('off')
plt.legend()
plt.show()
