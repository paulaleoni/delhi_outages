'''
this file is used for data exploration
'''

from pathlib import Path
import numpy as np
import pandas as pd


# Path
wd=Path.cwd()

# load data
data = pd.read_csv(wd.parent/'data'/'pollution'/'env_data_0726.csv')
location = pd.read_csv(wd.parent/'data'/'pollution'/'Indialocationlist.csv')


data['timearray'] = pd.to_datetime(data['timearray'])

data.describe()

stations = data.serialno.drop_duplicates().tolist()
# nearly 700 stations
len(stations) 

stations = pd.DataFrame(stations, columns=['station'])

# evaluate first and last reading
stations['first_reading'] = stations.apply(lambda row: data.loc[data['serialno'] == row.station, 'timearray'].min(), axis = 1)
stations['last_reading'] = stations.apply(lambda row: data.loc[data['serialno'] == row.station, 'timearray'].max(), axis = 1)
stations['time_coverage'] = stations.apply(lambda row: row.last_reading -row.first_reading, axis = 1)

# add dummies for pollution available
sensor = data['sensorName'].drop_duplicates().tolist()

for s in sensor:
    stations[s] = stations.apply(lambda row: True in data.loc[data['serialno'] == row.station, 'sensorName'].str.contains(s).tolist(), axis = 1)

# percent of missings for each stations in avrangearray
# missing = 0
stations['missing_perc'] = stations.apply(lambda row: data.loc[(data['serialno'] == row.station) & (data['avrangearray'] == 0), 'avrangearray'].count() / data.loc[data['serialno'] == row.station, 'avrangearray'].count(), axis=1)
# multiply by 100
stations['missing_perc'] = stations['missing_perc'] * 100

np.mean(stations.missing_perc) # 4% missing


# merge with Indialocationlist to get geo dimension
stations = stations.merge(location, left_on = 'station', right_on = 'serialNo').drop('serialNo', axis = 1)

# write to csv 
stations.to_csv(wd.parent/'data'/'data_transformed'/'stations_w_pol_loc.csv')