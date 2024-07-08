import requests
import json
from pimsys.regions.RegionsDb import RegionsDb
from shapely import geometry
from datetime import datetime
import pickle

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

def get_images(settings_db, interval, regions, model_id=None, get_model_images=False):
    all_image_ids = []
    all_images = []
    missed_regions = []

    with RegionsDb(settings['settings_db']) as database:
        for region in regions:
            coords = [region['bounds'].centroid.x, region['bounds'].centroid.y]
            images = database.get_optical_images_containing_point_in_period(coords, interval_utc)
#            print(images)
#            print(len(images))
#            print('type',  len(images))
            wms_images = sorted(images, key=lambda x: x["capture_timestamp"])
            wms_images = [x for x in wms_images if x["source"] != "Sentinel-2"]
            # wms_images = [x for x in wms_images if x['source'] == 'SkySat']
            for image in wms_images:
                if image["wms_layer_name"] not in all_image_ids:
                    all_image_ids.append(image["wms_layer_name"])
                    all_images.append(image)
    # Sort images based on classification
    return all_images


def get_utc_timestamp(x: datetime):
        return int((x - datetime(1970, 1, 1)).total_seconds())

customer_name = 'TC-Energy-Pilot'


#dates = [datetime(2023, 7, 1), datetime(2023, 7, 15)]

dates = [datetime(2023, 11, 23), datetime(2023, 12, 13)]

#source_list = [56]

database = RegionsDb(settings['settings_db'])
regions = database.get_regions_by_customer(customer_name)
#print('rrrrrrrrr', regions[0])
print('lenregions', len(regions)) #860 stathera!!
database.close()

# customer = Customer(customer_name, settings=settings)
interval_utc = [get_utc_timestamp(dates[0]), get_utc_timestamp(dates[1])]
layers_all = get_images(settings['settings_db'], interval_utc, regions)
layers_all = sorted(layers_all, key=lambda x: x["capture_timestamp"])

print(len(layers_all))


with open('Dec_basemap_regions.p', 'wb') as f:
     pickle.dump(layers_all, f)
