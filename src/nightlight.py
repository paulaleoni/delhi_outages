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
#from rasterio.transform import xy
import rioxarray as rxr
#from shapely.geometry.multipolygon import MultiPolygon
#from shapely import wkt
#import numpy as np
import rasterio
from rasterio.mask import mask
#import matplotlib.colors as colors
#from adjustText import adjust_text
#import urllib.request
#import io

#########################
# Functions
##########################

def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

################################

wd = Path.cwd()

shp_file = wd.parent/'data'/'shp'/'stanford-sh819zz8121-shapefile.zip'
grid_file = wd.parent/'data'/'grid'/'gis.zip'
tif_file = wd.parent/'data'/'satellite'/'SVDNB_npp_d20220112.rade9d.tif'

# load shapefile in geopandas dataframe 
shp = gpd.read_file(shp_file)
# keep only delhi
shp = shp.loc[shp.nam == 'DELHI',].reset_index(drop=True)
grid = gpd.read_file(grid_file)

grid.plot(edgecolor='grey', alpha = .5, color='None')

coords = getFeatures(shp)

url = 'https://eogdata.mines.edu/nighttime_light/nightly/rade9d/SVDNB_npp_d20220111.rade9d.tif'

test =  rxr.open_rasterio(tif_file, masked=True).rio.clip(coords, from_disk=True)
test.plot()
plt.show()

baseurl = 'https://eogdata.mines.edu/nighttime_light/'
username = "plbeck@web.de"
password = "dana@uni4"

'''
password_mgr = urllib.request.HTTPPasswordMgrWithDefaultRealm()
password_mgr.add_password(None, baseurl, username, password)

req = urllib.request.Request(url)
rxr.open_rasterio(url, masked=True, lock=False).rio.clip(coords, from_disk=True)

with urllib.request.urlopen(req) as resp:
    print(resp.read())
    raster = rxr.open_rasterio(io.BytesIO(resp.read()), masked=True).rio.clip(coords, from_disk=True)
    raster.plot()
    plt.show()
'''
################################
import requests
import json
import io
#import os
# Retrieve access token
params = {    
    'client_id': 'eogdata_oidc',
    'client_secret': '2677ad81-521b-4869-8480-6d05b9e57d48',
    'username': username,
    'password': password,
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
#response = requests.get(data_url, headers = headers, stream=True)
#response.encoding = 'utf-8'
#response.text

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



from PIL import Image
#test = Image.open(io.BytesIO(req.content))

import zipfile
import gzip
req = requests.get(data_url, headers = headers, stream=True)
with gzip.open('testfile.tif.gz', 'wb') as gz:
    gz.write(req.content)
with zipfile.ZipFile('testfile.zip', 'w',compression=zipfile.ZIP_DEFLATED) as out:
    out.write(req.content) 


'''

fig, ax = plt.subplots()
grid.plot(edgecolor='grey', alpha = .5, color='None', ax=ax)
test.plot(ax=ax)
plt.show()
'''

# make array to dataframe
test = rxr.open_rasterio('/vsigzip/testfile.tif.gz', masked=True).rio.clip(coords, from_disk=True)
df = test.to_dataframe(name = 'pixel')
df.reset_index(inplace=True)
df['point'] = df.apply(lambda row: Point(row.x, row.y), axis=1)
df['grid_id'] = np.nan

# add values to grid
grid['pixel_med'] = 0
for i in grid.index:
    geom = grid.geometry[i]
    id = grid.grid_id[i]
    bool =  df.apply(lambda row: row.point.within(geom), axis=1)
    med = df.pixel[bool].median()
    grid.loc[i,'pixel_med'] = med

fig, ax = plt.subplots()
grid.plot('pixel_med',edgecolor='grey', alpha = .5, cmap = 'RdPu', ax=ax)
plt.axis('off')
plt.show()
