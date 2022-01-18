'''
get nightly data from 'https://eogdata.mines.edu/nighttime_light/' and subset using the shapefile for Delhi

open questions:
- get data from url
- which time span
- ideally would get data at grid size level
- is there enough variation
- are clouds an issue, i.e. did the data correct for that or do we need a control (dummies provided)

'''

from pathlib import Path
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import datetime
#from rasterio.transform import xy
import rioxarray as rxr
#from shapely.geometry.multipolygon import MultiPolygon
#from shapely import wkt
#import numpy as np
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
#import matplotlib.colors as colors
#from adjustText import adjust_text
import requests
import json
import io
from hide import eogdata_user, eogdata_pw

#########################
# Functions
##########################

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]


def tif_from_url(date, shapefile):
    # shapefile coords
    coords = getFeatures(shapefile)

    #url corresponding to date
    url = f'https://eogdata.mines.edu/nighttime_light/nightly/rade9d/SVDNB_npp_d{date}.rade9d.tif'

    # Retrieve access token
    params = {    
        'client_id': 'eogdata_oidc',
        'client_secret': '2677ad81-521b-4869-8480-6d05b9e57d48',
        'username': eogdata_user,
        'password': eogdata_pw,
        'grant_type': 'password'
    }
    token_url = 'https://eogauth.mines.edu/auth/realms/master/protocol/openid-connect/token'
    response = requests.post(token_url, data = params)
    access_token_dict = json.loads(response.text)
    access_token = access_token_dict.get('access_token')
    # Submit request with token bearer
    ## Change data_url variable to the file you want to download
    data_url = url
    auth = 'Bearer ' + access_token
    headers = {'Authorization' : auth}
    # make request
    req = requests.get(data_url, headers = headers, stream=True)
    with MemoryFile(req.content) as memfile:
        with memfile.open() as data:
            tif = rxr.open_rasterio(data, masked=True).rio.clip(coords, from_disk=True)
    return tif

def dates_list(start, end, step=1):
    start = datetime.date(int(start[0:4]), int(start[4:6]), int(start[6:8]))
    end = datetime.date(int(end[0:4]), int(end[4:6]), int(end[6:8]))
    list = []
    day = start
    while day <= end:
        list.append(day)
        day += datetime.timedelta(step)
    return list


###########################
# shapefiles
##########################

wd = Path.cwd()

shp_file = wd.parent/'data'/'shp'/'stanford-sh819zz8121-shapefile.zip'
grid_file = wd.parent/'data'/'grid'/'gis.zip'

# load shapefile in geopandas dataframe 
shp = gpd.read_file(shp_file)
# keep only delhi
shp = shp.loc[shp.nam == 'DELHI',].reset_index(drop=True)
grid = gpd.read_file(grid_file)


###########################
# get tif 
#########################

# define date to retrive tif
year = '2022'
month = '01'
day = '11'
start_date = year + month + day
end_date = year + month + '12'

# list of dates
dates = dates_list(start_date, end_date)

for d in dates:
    string = d.strftime('%Y%m%d')
    # get data 
    data = tif_from_url(string, shp)
    df = data.to_dataframe(name = 'pixel')
    df.reset_index(inplace=True)
    df['point'] = df.apply(lambda row: Point(row.x, row.y), axis=1)
    # add values to grid
    grid[f'pixel_med_{string}'] = 0
    for i in grid.index:
        geom = grid.geometry[i]
        #id = grid.grid_id[i]
        bool =  df.apply(lambda row: row.point.within(geom), axis=1)
        med = df.pixel[bool].median()
        grid.loc[i,f'pixel_med_{string}'] = med

fig, ax = plt.subplots()
grid.plot(f'pixel_med_{end_date}',edgecolor='grey', alpha = .5, cmap = 'RdPu', ax=ax)
plt.axis('off')
plt.show()


'''
resp = requests.get(data_url, headers = headers, stream=True)
#test = rxr.open_rasterio(resp.content, masked=True, chunks=True).rio.clip(coords)

with requests.get(data_url, headers = headers, stream=True) as resp:
    pt = rasterio.path.parse_path(resp.url)
    print(pt)
    print(resp.url)
    test = rxr.open_rasterio(resp.content,masked=True, chunks=True).rio.clip(coords)
    #print(test.meta)
    print(resp.headers.get('content-type'))
    print(resp.url)
    #print(resp.json())
    print(resp.history)
    #print(resp.raw.read(10))
    #resp.encoding = 'utf-16'
    print(resp.encoding)
    print(resp.headers)
    #dict = resp.json()
    #print(resp.json())
    bytes = io.BytesIO(resp.content)
    #print(bytes)
    test = rxr.open_rasterio(bytes, masked=True).rio.clip(coords, from_disk=True)
    print(test.meta)
    print(test)
    test.plot()
    plt.show()
'''
'''
fig, ax = plt.subplots()
grid.plot(edgecolor='grey', alpha = .5, color='None', ax=ax)
test.plot(ax=ax)
plt.show()
'''
'''
with gzip.open('testfile.tif.gz', 'wb') as gz:
    gz.write(req.content)
'''