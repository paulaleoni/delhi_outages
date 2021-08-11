'''
use pollution data from DANA_data
'''

from pathlib import Path
import pandas as pd
import zipfile
import datetime
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np

# Path
wd=Path.cwd()
parent = wd.parent


#*#########################
#! LOAD DATA
#*#########################

# map of India
map = gpd.read_file(wd.parent/'data'/'shp'/'sh819zz8121.shp')
# delhi
delhi = {'city':'delhi', 'geometry':Point(77.23149, 28.65195)}
# source: https://www.geodatos.net/en/coordinates/india/delhi
delhi = gpd.GeoDataFrame(delhi, index=[0], crs='EPSG:4326')

grid = gpd.read_file(wd.parent/'data'/'grid'/'gridnwq.shp')


# list of zips
zips = ['Delhi_data2.zip','DANA_data1.zip', 'DelhiData1.zip']
# list of folders in zips, order must match
folders = ['DANA_data', 'DANA_data1', 'DANA_data']

# empty dataframe with all stations
stations = pd.DataFrame(columns=['uid','stationname','lat','lon'])
# empty dataframe for the actual data
data = pd.DataFrame(columns=['station_id', 'DateTime', 'AQI-IN','aqi','h','pm10','pm25','pm1','t'])
for i in range(len(zips)):
    # open zip
    zip_path = wd.parent/'data'/'pollution'/zips[i]
    zip = zipfile.ZipFile(zip_path)
    folder = folders[i]
    # read stationlist
    stat = pd.read_csv(zip.open(folder + '/' + 'Stationlist.csv'))
    # extract all station id's
    stat_uid = stat.uid.drop_duplicates().tolist()
    # append to stations dataframe
    stations = stations.append(stat, ignore_index=True)
    # now go through all station id's
    for file in stat_uid:
        # read file
        path = folder + '/' + file + '_April2021toMarch2020sensorhistory.csv'
        df = pd.read_csv(zip.open(path), header=None, names=['DateTime', 'AQI-IN','aqi','h','pm10','pm25','pm1','t'], sep=',') 
        # drop first rows with '<b' etc
        df = df[df.apply(lambda row: '<b' not in row['DateTime'], axis=1)]
        df = df.reset_index(drop=True)
        # extract to actual columns and define as new column names
        headers =df.iloc[0]
        df.columns = headers
        # drop row with column names
        df = df.drop(0)
        # drop NaN columns
        df = df.dropna(axis=1, how='all')
        # make to datetime
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        # add a column with station id
        df['station_id'] = file
        # append to dataframe
        data = data.append(df, ignore_index=True)

# drop duplicated rows
stations = stations.drop_duplicates()
data = data.drop_duplicates()

# rename uid
stations = stations.rename(columns={'uid':'station'})

# pollution data from str to float
for s in data.columns[2:]:
    data.loc[:,s] = data.loc[:,s].astype(float)

# export to csv
data.to_csv(wd.parent/'data'/'data_transformed'/'pollution_data_02.csv')

#*#########################
#! Stations Information
#*#########################

# add first and last reading and time coverage in days
stations['first_reading'] = stations.apply(lambda row: data.loc[data['station_id'] == row.station, 'DateTime'].min(), axis = 1)
stations['last_reading'] = stations.apply(lambda row: data.loc[data['station_id'] == row.station, 'DateTime'].max(), axis = 1)
stations['time_coverage'] = stations.last_reading - stations.first_reading

# average time coverage is nearyl 500 days
sum(stations.time_coverage[0:500].tolist(), datetime.timedelta()) /len(stations.time_coverage[0:500])

# add information of pollutants
for s in data.columns[2:]:
    stations[s] = stations.apply(lambda row: np.mean(data.loc[data.station_id == row.station,s]), axis=1)

# add column geometry
stations['geometry'] = stations.apply(lambda row: Point(row.lon, row.lat), axis=1)

# make a geo df
stations = gpd.GeoDataFrame(stations)

## add grid id
dict = {}
for stat in stations.station: 
    for cell in grid.grid_id:
        # check if stat in cell
        point = stations.loc[stations.station == stat, 'geometry'].reset_index(drop=True)[0]
        poly = grid.loc[grid.grid_id == cell,'geometry'].reset_index(drop=True)[0]
        bool = point.within(poly)
        if bool == True:
            # dict with station id as key and values polygon and grid id
            dict[stat] = (poly,cell)
        else: dict[stat] = (np.nan,np.nan) 

stations['grid_id'] = stations.apply(lambda row: dict[row.station][1], axis=1)
stations['grid_geometry'] = stations.apply(lambda row: dict[row.loc['station']][0], axis=1)


# export to csv
stations.to_csv(wd.parent/'data'/'data_transformed'/'stations_02.csv')

#*#########################
#! grid
#*#########################

grid['center'] = grid.apply(lambda row: row.geometry.centroid,axis=1)

# add stations column to grid
dict = {}
for cell in grid.grid_id:
    list = []
    for stat in stations.station:
        point = stations.loc[stations.station == stat, 'geometry'].reset_index(drop=True)[0]
        poly = grid.loc[grid.grid_id == cell,'geometry'].reset_index(drop=True)[0]
        bool = point.within(poly)
        if bool == True:
            list.append(stat)
    dict[cell] = list        

grid['stations'] = grid.apply(lambda row: dict[row.grid_id], axis=1)

### add values for sensor based on stations in each gridcell
''' first only make simple average, later do weighting'''
# initialize empty columns
for s in data.columns[2:]:
    grid[f'{s}_value'] = 0


for gr in grid.grid_id:
    # extract list of stations in grid cell
    stat = grid.loc[grid.grid_id == gr,'stations'].reset_index(drop=True)[0]
    # if stat is not empty
    if len(stat) > 0:
        # for each pollutant
        for s in data.columns[2:]:
            # set averag to 0
            avg = 0
            # compute average of each stations and add those up
            for st in stat:
                values = data.loc[data.station_id == st,s]
                avg += np.mean(values)
            # devide by the number of stations
            avg = avg / len(stat)
            grid.loc[grid.grid_id == gr,f'{s}_value'] = avg

# export to csv
grid.to_csv(wd.parent/'data'/'data_transformed'/'grid_w_stat_02.csv')

#*#########################
#! map
#*#########################


# extract polygon of delhi in map
point_delhi = delhi['geometry'][0]
in_delhi = map.apply(lambda row: point_delhi.within(row.geometry), axis=1)
map_delhi = map[in_delhi].reset_index(drop=True)

value = 'pm25_value'
fig, ax = plt.subplots(figsize=(12,12))
map_delhi.plot(color='white', edgecolor='blue', ax=ax)
grid.plot(value,  edgecolor='grey',alpha=.5, cmap = 'RdPu', legend = True,ax=ax)
stations.plot(color = 'darkred', ax=ax, label='stations', alpha = .5)
plt.title(f'Delhi stations with {value}')
plt.text(77.4,28.85,f'{len(stations)} stations')
plt.axis('off')
plt.legend()
plt.show()
fig.savefig('visualizations/Delhi_stat_02.png')

