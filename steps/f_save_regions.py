import rasterio
import rasterio.features
from pimsys.regions.RegionsDb import RegionsDb
import pyproj
from shapely import geometry, ops
from shapely.geometry import box
import numpy as np
import cv2
import pandas as pd
import pickle
from PIL import Image
from geopy.distance import geodesic
import requests
import os


settings = {"settings_client": {
     "user": "root",
     "password": "9YxeCg4R2Un%",
     "host": "tcenergy.orbitaleye.nl",
     "port": 9949
     },
    "settings_db" : {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
        }
    }



def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    # Open the input TIF file
#    image = np.expand_dims(image, 2)

    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename + '.tif', 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
#    print('shape', image.shape)
    for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
#    new_dataset.write(image, indexes=1)
    new_dataset.close()

    return True

import cv2
import numpy as np
import requests
from geopy.distance import geodesic

from pimsystools.PimsysClient import PimsysClient
client = PimsysClient('tcenergy.orbitaleye.nl', 9949)
client.login('root', '9YxeCg4R2Un%')
overlays = client.getOverlays({})

print(overlays)

def get_image_from_overlay(overlay, region_bounds):

    layer_name = overlay['name']
    # Define layer name
    pixel_resolution_x = 0.5#layer["pixel_resolution_x"]
    pixel_resolution_y = 0.5#layer["pixel_resolution_y"]

    region_width = geodesic(
    (region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[1], region_bounds.bounds[2])
    ).meters
    region_height = geodesic(
    (region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[3], region_bounds.bounds[0])
    ).meters

    width = int(round(region_width / pixel_resolution_x))
    height = int(round(region_height / pixel_resolution_y))

    arguments = {
        'layer_name': layer_name,
        'bbox': '%s,%s,%s,%s' % (region_bounds.bounds[0], region_bounds.bounds[1], region_bounds.bounds[2], region_bounds.bounds[3]),
        'width': width,
        'height': height
    }

    # get URL
    if 'url' in overlay.keys():
         url = overlay['url'][28:] + "&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments


    print(url)
    resp = requests.get(url, auth=('mapserver_user', 'tL3n3uoPnD8QYiph'))

    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]

    return image

customer_name = 'TC-Energy-Pilot'
database = RegionsDb(settings['settings_db'])
regions = database.get_regions_by_customer(customer_name)
print('lenregions', len(regions))
database.close()



#layers_all = pd.read_pickle('June_basemap_regions.p')
#layers_all = sorted(layers_all, key=lambda x: x["capture_timestamp"])
##layers_all.reverse()
##layers_all = layers_all[0:1]
#print(len(layers_all))

#cnt = 0

all_region_bounds = []

existed_ids = os.listdir('./REGIONS_Dec/')

cnt=0
for region in regions:
  print('{}/{}'.format(cnt, len(regions)))
#  print(region)
  imfile = '{}_{}.tif'.format(str(region['id']), str(region['region_customer_id']))
  print(imfile)
  if imfile not in existed_ids:
    try:
     img = get_image_from_overlay(overlays[0], region['bounds'])
     print(img.shape)

#        img = get_image_from_layer(layer_image, region_image['bounds'])
#        print('img', img.shape)
#        img, img_norm = np.transpose(img, (1,2,0)), np.transpose(img_norm, (1,2,0))
     save_tif_coregistered('./REGIONS_Dec/{}_{}'.format(str(region['id']), str(region['region_customer_id'])), img, region['bounds'], channels = 3)
    except:
      pass
  cnt = cnt+1
