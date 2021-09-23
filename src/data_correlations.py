'''
check temporal correlation correlation in pollutants between the AQI data and government official monitor data

!!! consider delhi only

source: 
env_data_0726.zip (AQI data)
gov_monitor_data.zip (government official monitor data)
'''

from pathlib import Path
import zipfile
import pandas as pd
import geopandas as gpd
from shapely import geometry
from shapely.geometry import Point
from haversine import haversine
import numpy as np
import matplotlib.pyplot as plt
import datetime
import random

# Path
wd=Path.cwd()

#*#########################
#! FUNCTIONS
#*#########################

def time_overlap(start1, start2, end1, end2):
    '''
    inputs are datetime objects
    returns date range if there is an overlap, otherwise NAN
    '''
    # get the time range
    if start1 > start2: start = start1
    else: start = start2
    if end1 < end2: end = end1
    else: end = end2
    if start < end: return pd.date_range(start = start, end=end)
    else: return np.nan

#*#########################
#! LOAD DATA
#*#########################

map = gpd.read_file(wd.parent/'data'/'shp'/'sh819zz8121.shp')
delhi = {'city':'delhi', 'geometry':Point(77.23149, 28.65195)}
# source: https://www.geodatos.net/en/coordinates/india/delhi
delhi = gpd.GeoDataFrame(delhi, index=[0], crs='EPSG:4326')

point_delhi = delhi['geometry'][0]
in_delhi = map.apply(lambda row: point_delhi.within(row.geometry), axis=1)
map_delhi = map[in_delhi].reset_index(drop=True)

##############################################

zip = zipfile.ZipFile(wd.parent/'data'/'pollution'/'env_data_0726.zip')
data_aqi = pd.read_csv(zip.open('env_data_0726.csv'))
aqi_loc = pd.read_csv(wd.parent/'data'/'pollution'/'Indialocationlist.csv')

# merge
data_aqi = data_aqi.merge(aqi_loc, left_on = 'serialno', right_on = 'serialNo').drop(['serialNo', 'locationname', 'cityName', 'stateName', 'countryName'], axis = 1)

# make column geometry
data_aqi['geometry'] = data_aqi.apply(lambda row: Point(row.lon, row.lat), axis = 1)

### filter directly to save memory space 
# unique aqi monitors
monitors = data_aqi[['serialno', 'geometry']].drop_duplicates('serialno').reset_index(drop=True)

# filter for monitors in delhi
delhi = monitors.apply(lambda row: row.geometry.within(map_delhi.geometry[0]), axis=1)
monitors = monitors[delhi]

# filter data_aqi
monlist = monitors.serialno.drop_duplicates().tolist()
filter =  data_aqi.apply(lambda row: row.serialno in monlist , axis = 1)
data_aqi = data_aqi[filter]

#data_aqi = data_aqi[(data_aqi.sensorName == 'PM25') & (data_aqi.timearray.dt.date == pd.Timestamp('2019-01-24'))][0:500]

#########################################

zip = zipfile.ZipFile(wd.parent/'data'/'pollution'/'gov_monitor_data.zip')
data_gov = pd.read_stata(zip.open('mergedpollution_Nov2017Dec2019.dta'))
gov_loc = pd.read_csv(zip.open('stations_coordinates_allIndia.csv'))

# merge
data_gov = data_gov.merge(gov_loc, left_on = 'location', right_on = 'location').drop(['city', 'attribution'], axis = 1)


# make geometry column
data_gov['geometry'] = data_gov.apply(lambda row: Point(row.longitude, row.latitude), axis = 1)



#*#########################
#! Find closest points
#*#########################

# add gov monitor information

monitors['closest_gov_mon'] = np.nan
monitors['closest_gov_mon_point'] = np.nan
monitors['closest_gov_mon_dist'] = np.nan

# get unique gov monitors
gov_loc = data_gov[['location', 'geometry']].drop_duplicates('location').reset_index(drop=True)

for g in monitors.index:
    # get geometry aqi
    g_loc = (monitors.geometry[g].x, monitors.geometry[g].y)
    dist = {}
    for d in gov_loc.index:
        # get geometry of gov monitors
        d_loc = (gov_loc.geometry[d].x, gov_loc.geometry[d].y)
        # calculate distance
        distance = haversine(g_loc,d_loc)
        dist[d] = distance
    # find minimum distance
    mindist = min(dist.values())
    minpoint = [key for key in dist if dist[key] == mindist]
    # add to data
    monitors.loc[g,'closest_gov_mon'] = gov_loc.loc[minpoint[0], 'location']
    monitors.loc[g,'closest_gov_mon_point'] = gov_loc.loc[minpoint[0], 'geometry']
    monitors.loc[g,'closest_gov_mon_dist'] = mindist



# filter data_gov
monlist = monitors.closest_gov_mon.drop_duplicates().tolist()
filter =  data_gov.apply(lambda row: row.location in monlist, axis = 1)
data_gov = data_gov[filter]

# distance is less than 1 km on average, 3km is max
monitors.closest_gov_mon_dist.describe() 


#*#########################
#! Prepare Data
#*#########################


# convert times
data_aqi['timearray'] = pd.to_datetime(data_aqi['timearray'])
data_gov['fromdate'] = pd.to_datetime(data_gov['fromdate'])

#extract first and last readings
monitors['first_reading'] = monitors.apply(lambda row: data_aqi.loc[data_aqi['serialno'] == row.serialno, 'timearray'].min(), axis = 1)
monitors['last_reading'] = monitors.apply(lambda row: data_aqi.loc[data_aqi['serialno'] == row.serialno, 'timearray'].max(), axis = 1)

monitors['gov_first_reading'] = monitors.apply(lambda row: data_gov.loc[data_gov['location'] == row.closest_gov_mon, 'fromdate'].min(), axis = 1)
monitors['gov_last_reading'] = monitors.apply(lambda row: data_gov.loc[data_gov['location'] == row.closest_gov_mon, 'fromdate'].max(), axis = 1)

# get overlap from function time_overlap
monitors['time_overlap'] = monitors.apply(lambda row: time_overlap(row.first_reading, row.gov_first_reading, row.last_reading, row.gov_last_reading), axis = 1)

monitors.to_csv(wd.parent/'data'/'data_transformed'/'monitor_matching.csv')

#*#########################
#! Correlations
#*#########################

'''
aqi-data is measured every hour
gov monitors measure every 15 min -> filter for those at XX:00

first only consider PM25 -> filter data_aqi
make a new column in data_aqi -> PM25 for same hour from nearest gov station
'''

# data_aqi['gov_values'] = np.nan # could also loop over sensors
#dict = {'PM25': 'pm25', 'PM10': 'pm10'}

random_stat = random.choices(data_aqi.serialno.drop_duplicates().tolist(), k = 20)
#data_aqi = data_aqi[(data_aqi.sensorName == 'PM25') & (data_aqi.timearray.dt.date == pd.Timestamp('2019-01-24')) & data_aqi.serialno.isin(random_stat)]
data_aqi['gov_pm25'] = np.nan
for i in data_aqi[data_aqi.sensorName == 'PM25'].index:
    #i = data_aqi[data_aqi.sensorName == 'PM25'].index[0]
    # get aqi monitor
    stat = data_aqi.loc[i,'serialno']
    # get date and extract date-range
    date = data_aqi.timearray.dt.date[i]
    range = monitors.time_overlap[monitors.serialno == stat].reset_index(drop=True)[0].date
    # check if date in time_overlap
    if date in range:
        # get nearest gov monitor
        gov = monitors.closest_gov_mon[monitors.serialno == stat].reset_index(drop=True)[0]
        # get time
        time = data_aqi.timearray[i].hour
        # filter for pm25 for nearest monitor, same time, at full hour (XX:00)
        filter = (data_gov.location == gov) & (data_gov.fromdate.dt.date == date) & (data_gov.fromdate.dt.hour == time) & (data_gov.fromdate.dt.minute == 00)
        if len(data_gov.loc[filter, 'pm25']) > 0:
            # extract series
            pm = data_gov.loc[filter, 'pm25']
            # filter for non-empty values
            pm = pm[pm != 0].reset_index(drop = True)[0]
            # overwrite column
            data_aqi.loc[i,'gov_pm25'] = pm


data_aqi['gov_pm25'] = data_aqi['gov_pm25'].astype(float)
station = data_aqi.loc[:,'serialno'].drop_duplicates().tolist()

#d = data_aqi.loc[data_aqi.sensorName == 'PM25',]
for i in station:
    d = data_aqi.loc[data_aqi.serialno == i,]
    c = np.corrcoef(d.avrangearray,d.gov_pm25)[0,1]
    # plot both values
    fig, ax = plt.subplots()
    plt.plot(d.timearray, d.avrangearray, label='aqi', color = 'blue')
    plt.plot(d.timearray, d.gov_pm25, label='gov', color = 'red')
    plt.legend()
    plt.suptitle(f'correlation: {c:.2f}; station: {i}' )
    plt.show()
    p  = 'correlations/corr_' + str(i)
    fig.savefig(p)



#*#########################
#! Maps
#*#########################


monitors_delhi = gpd.GeoDataFrame(monitors)


gov_delhi = gpd.GeoDataFrame(monitors, geometry='closest_gov_mon_point')


fig, ax = plt.subplots(figsize=(12,12))
map_delhi.plot(color='white', edgecolor='grey', ax=ax)
monitors_delhi.plot(color='darkred',markersize=10, ax=ax, label='aqi_data')
gov_delhi.plot(color='darkblue',markersize=10, ax=ax, label='gov monitors')
plt.axis('off')
plt.legend()
plt.show()
fig.savefig('monitors_gov_aqi')



