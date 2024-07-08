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


def get_image_from_layer(layer, region_bounds):
    layer_name = layer['wms_layer_name']

    # Define layer name
    pixel_resolution_x = layer["pixel_resolution_x"]
    pixel_resolution_y = layer["pixel_resolution_y"]

    region_width = geodesic((region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[1], region_bounds.bounds[2])).meters
    region_height = geodesic((region_bounds.bounds[1], region_bounds.bounds[0]), (region_bounds.bounds[3], region_bounds.bounds[0])).meters

    width = int(round(region_width / pixel_resolution_x))
    height = int(round(region_height / pixel_resolution_y))

    arguments = {
        'layer_name': layer_name,
        'bbox': '%s,%s,%s,%s' % (region_bounds.bounds[0], region_bounds.bounds[1], region_bounds.bounds[2], region_bounds.bounds[3]),
        'width': width,
        'height': height
    }

    # get URL
    if 'image_url' in layer.keys():
        if layer['downloader'] == 'geoserve':
            arguments['bbox'] = '%s,%s,%s,%s' % (region_bounds[1], region_bounds[0], region_bounds[3], region_bounds[2])
            url = layer['image_url'] + "&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&CRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments
        elif layer['downloader'] == 'sentinelhub':
            url = layer['image_url']
        else:
            url = layer['image_url'] + "&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments
    else:
        url = "https://maps.orbitaleye.nl/mapserver/?map=/maps/_%(layer_name)s.map&VERSION=1.0.0&SERVICE=WMS&REQUEST=GetMap&STYLES=&SRS=epsg:4326&BBOX=%(bbox)s&WIDTH=%(width)s&HEIGHT=%(height)s&FORMAT=image/png&LAYERS=%(layer_name)s" % arguments

    if layer['downloader'] == 'geoserve':
        resp = requests.get(url, auth=('ptt', 'yOju6YLPK6Pnqm2C'))
    else:
        resp = requests.get(url, auth=('mapserver_user', 'tL3n3uoPnD8QYiph'))

    image = np.asarray(bytearray(resp.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]

    return image


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


def normalise_bands(image, percentile_min=2, percentile_max=98):
    # Get and normalize tif tiles
    tmp = []
    for i in range(image.shape[0]):
        image_band = image[i, :, :].astype(np.float32)
        min_val = np.nanmin(image_band[np.nonzero(image_band)])
        max_val = np.nanmax(image_band[np.nonzero(image_band)])
        image_band = (image_band - min_val) / (max_val - min_val)
        image_band_valid = image_band[np.logical_and(image_band > 0, image_band < 1.0)]
        perc_2 = np.nanpercentile(image_band_valid, percentile_min)
        perc_98 = np.nanpercentile(image_band_valid, percentile_max)
        band = (image_band - perc_2) / (perc_98 - perc_2)
        band[band < 0] = 0.0
        band[band > 1] = 1.0
        tmp.append(band)
    return np.array(tmp).astype(np.float32)

def load_tif_image_window(layer, poly_bounds, norm=True):
    # Define mapserver path
    mapserver_path = "/cephfs/mapserver/data"
    # Get path to after image
    layer_datetime = layer["capture_timestamp"]
    path_tif = (
        "/".join(
            [
                mapserver_path,
                layer_datetime.strftime("%Y%m%d"),
                layer["wms_layer_name"],
            ]
        )
        + ".tif"
    )
    # Load image
    tif = rasterio.open(path_tif)
    # Define crs transform
    proj = pyproj.Transformer.from_crs(4326, str(tif.crs), always_xy=True)
    bounds = ops.transform(proj.transform, poly_bounds)
    bounds_window = rasterio.features.bounds(bounds)
    # Define window
    window = rasterio.windows.from_bounds(*bounds_window, tif.transform)
    # Read image array
    img = tif.read(window=window)
    transform_image = tif.transform
    # Normalize bands on min and max
    if norm:
        img_norm = normalise_bands(img)
    else:
        if img.dtype == "uint8":
            img_norm = img / 255.0
        else:
            img_norm = img

    tif.close()
    return img, img_norm, tif.transform


customer_name = 'TC-Energy-Pilot'
database = RegionsDb(settings['settings_db'])
regions = database.get_regions_by_customer(customer_name)
print('lenregions', len(regions))
database.close()

layers_all = pd.read_pickle('Dec_basemap_regions.p')
layers_all = sorted(layers_all, key=lambda x: x["capture_timestamp"])
#layers_all.reverse()
#layers_all = layers_all[0:1]
print(len(layers_all))

cnt = 0

all_region_bounds = []

for layer_image in layers_all:
    print('{}/{}'.format(cnt,len(layers_all)))
    regions_image = []
    for region in regions:
      if region['bounds'].intersects(layer_image['valid_area']):
        inter = region['bounds'].intersection(layer_image['valid_area'])
        region_area = region['bounds'].area
        inter_perc = (inter.area/region_area)*100
        if inter_perc>50:
          print('inter', inter_perc)
          regions_image.append(region)

#    regions_image = [region for region in regions if region['bounds'].intersects(layer_image['valid_area'])]
    cnt = cnt + 1
    for region_image in regions_image:
      if region_image['bounds'] not in all_region_bounds:
        all_region_bounds.append(region_image['bounds'])
#        img, img_norm, _ = load_tif_image_window(layer_image, region_image['bounds'])
        img = get_image_from_layer(layer_image, region_image['bounds'])
        print('img', img.shape)
#        img, img_norm = np.transpose(img, (1,2,0)), np.transpose(img_norm, (1,2,0))
##        if img_norm.shape[0]>=p and img_norm.shape[1]>=p:
#        whole_image = img_norm*255
##        print(whole_image.shape)

#        whole_image = whole_image[:,:,0:3]
        save_tif_coregistered('./REGIONS_Dec23/{}_{}'.format(str(region_image['id']), str(region_image['region_customer_id'])), img, region_image['bounds'], channels = 3)
      else:
        print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
#        save_tif_coregistered('./REGIONS_June_01_15/{}_{}'.format(str(region_image['id']), str(region_image['region_customer_id'])), whole_image, region_image['bounds'], channels = 3)


'''
missed_bounds = []
for region in regions:
  if region['bounds'] not in all_region_bounds:
    missed_bounds.append(region)

print('Missed bounds: ')
print(len(missed_bounds))


with open('missed_regions.p', 'wb') as f:
     pickle.dump(missed_bounds, f)
'''



