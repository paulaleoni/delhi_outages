'''
this file is used for data exploration of the pollution data
'''

from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import datetime
import zipfile

# Path
wd=Path.cwd()

#*#########################
#! LOAD DATA
#*#########################
zip = zipfile.ZipFile(wd.parent/'data'/'pollution'/'env_data_0726.zip')
data = pd.read_csv(zip.open('env_data_0726.csv'))
location = pd.read_csv(wd.parent/'data'/'pollution'/'Indialocationlist.csv')
grid = gpd.read_file(wd.parent/'data'/'grid'/'gridnwq.shp')

delhi = {'city':'delhi', 'geometry':Point(77.23149, 28.65195)}
# source: https://www.geodatos.net/en/coordinates/india/delhi
delhi = gpd.GeoDataFrame(delhi, index=[0], crs='EPSG:4326')

# map of India
map = gpd.read_file(wd.parent/'data'/'shp'/'sh819zz8121.shp')

#*#########################
#! Stations Information
#*#########################

# make timearray to datetimeindex datatype
data['timearray'] = pd.to_datetime(data['timearray'])

data.describe()

# list of all stations
stations = data.serialno.drop_duplicates().tolist()
# nearly 700 stations
len(stations) 

# initialize df with stations as rows
stations = pd.DataFrame(stations, columns=['station'])

# evaluate first and last reading, add to df
stations['first_reading'] = stations.apply(lambda row: data.loc[data['serialno'] == row.station, 'timearray'].min(), axis = 1)
stations['last_reading'] = stations.apply(lambda row: data.loc[data['serialno'] == row.station, 'timearray'].max(), axis = 1)
stations['time_coverage'] = stations.last_reading - stations.first_reading

# average time coverage # 200 days
sum(stations.time_coverage[0:500].tolist(), datetime.timedelta()) /len(stations.time_coverage[0:500])


# add dummies for pollution available
sensor = data['sensorName'].drop_duplicates().tolist()
for s in sensor:
    stations[s] = stations.apply(lambda row: True in data.loc[data['serialno'] == row.station, 'sensorName'].str.contains(s).tolist(), axis = 1)


# percent of missings for each stations in avrangearray
stations['missing_perc'] = stations.apply(lambda row: len(data.loc[(data['serialno'] == row.station) & (data['avrangearray'] == 0), 'avrangearray']) / len(data.loc[data['serialno'] == row.station, 'avrangearray']), axis = 1)

# multiply by 100
stations['missing_perc'] = stations['missing_perc'] * 100

''' assumes that 0 means missing'''

np.mean(stations.missing_perc) # 1.4% missing

#*#########################
#! geo dimension
#*#########################

# merge with Indialocationlist to get geo dimension
stations = stations.merge(location, left_on = 'station', right_on = 'serialNo').drop('serialNo', axis = 1)

# add column geometry
stations['geometry'] = stations.apply(lambda row: Point(row.lon, row.lat), axis=1)

#*#########################
#! extract delhi and add grid
#*#########################

# extract polygon of delhi in map
point_delhi = delhi['geometry'][0]
in_delhi = map.apply(lambda row: point_delhi.within(row.geometry), axis=1)
map_delhi = map[in_delhi].reset_index(drop=True)

# stations inside polygon of delhi
stations_delhi = stations.apply(lambda row: row.geometry.within(map_delhi.geometry[0]), axis=1)
stations_delhi = stations[stations_delhi]

np.mean(stations_delhi.missing_perc) # 1%


# assign gridcell to stations in dehli

# initialize dictionary
dict = {}
for stat in stations_delhi.station:
    for cell in grid.grid_id:
        # check if stat in cell
        point = stations_delhi.loc[stations_delhi.station == stat, 'geometry'].reset_index(drop=True)[0]
        poly = grid.loc[grid.grid_id == cell,'geometry'].reset_index(drop=True)[0]
        bool = point.within(poly)
        if bool == True:
            # dict with station id as key and values polygon and grid id
            dict[stat] = (poly,cell)

# add to stations_delhi
stations_delhi['grid_id'] = stations_delhi.apply(lambda row: dict[row.loc['station']][1], axis=1)
stations_delhi['grid_geometry'] = stations_delhi.apply(lambda row: dict[row.loc['station']][0], axis=1)


#*#########################
#! GRID as cross-section
#*#########################

# extract centroid of each grid
grid['center'] = grid.apply(lambda row: row.geometry.centroid,axis=1)

# add stations column to grid
dict = {}
for cell in grid.grid_id:
    list = []
    for stat in stations_delhi.station:
        point = stations_delhi.loc[stations_delhi.station == stat, 'geometry'].reset_index(drop=True)[0]
        poly = grid.loc[grid.grid_id == cell,'geometry'].reset_index(drop=True)[0]
        bool = point.within(poly)
        if bool == True:
            list.append(stat)
    dict[cell] = list        

grid['stations'] = grid.apply(lambda row: dict[row.grid_id], axis=1)

### add values for sensor based on stations in each gridcell
''' first only make simple average, later do weighting'''
# initialize empty columns
for s in sensor:
    grid[f'{s}_value'] = 0

#gr = 417
for gr in grid.grid_id:
    # extract list of stations in grid cell
    stat = grid.loc[grid.grid_id == gr,'stations'].reset_index(drop=True)[0]
    # centroid of grid cell
    #center = grid.loc[grid.grid_id == gr, 'center'].reset_index(drop=True)[0]
    # if stat is not empty
    if len(stat) > 0:
        # for each sensor calculate the average of stations in grid
        for s in sensor:
            avg = 0
            #s = sensor[0]
            #st = stat[0]
            for st in stat:
                # values
                x = data.loc[(data.serialno == st) & (data.sensorName == s),'avrangearray']
                # location of station
                #st_loc = stations_delhi.loc[stations_delhi.station == st, 'geometry'].reset_index(drop=True)[0]
                # distance between station and center of grid
                #dist = center.distance(st_loc)
                # average across time dimension of each station weighted by dist
                avg += np.mean(x) #/ dist
            # average of stations    
            avg = avg / len(stat)
            grid.loc[grid.grid_id == gr,f'{s}_value'] = avg      


#*#########################
#! EXPORT to csv
#*#########################

stations.to_csv(wd.parent/'data'/'data_transformed'/'stations_w_pol_loc.csv')
stations_delhi.to_csv(wd.parent/'data'/'data_transformed'/'stations_delhi.csv')
grid.to_csv(wd.parent/'data'/'data_transformed'/'grid_w_stat.csv')