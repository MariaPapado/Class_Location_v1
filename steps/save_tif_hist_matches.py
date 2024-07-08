import rasterio
import cv2
import numpy as np
from PIL import Image
import os
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import imutils
from tqdm import tqdm
from shapely import geometry

def check_shape_and_resize(raster_before, img_after):
    h_before, w_before = raster_before.height, raster_before.width
    h_after, w_after = img_after.shape[0], img_after.shape[1]
    img_before = raster_before.read()
#    img_after = raster_after.read()
    img_before = np.transpose(img_before, (1,2,0))
#    print(img_before.shape, img_after.shape)
    if h_before!=h_after or w_before!=w_after:
        img_after = cv2.resize(img_after, (w_before, h_before), cv2.INTER_NEAREST)
        return img_before, img_after
    else:
        return img_before, img_after



def save_tif_coregistered(filename, image, poly, channels=1, factor=1):
    height, width = image.shape[0], image.shape[1]
    geotiff_transform = rasterio.transform.from_bounds(poly.bounds[0], poly.bounds[1],
                                                       poly.bounds[2], poly.bounds[3],
                                                       width/factor, height/factor)

    new_dataset = rasterio.open(filename, 'w', driver='GTiff',
                                height=height/factor, width=width/factor,
                                count=channels, dtype='uint8',
                                crs='+proj=latlong',
                                transform=geotiff_transform)

    # Write bands
    if channels>1:
     for ch in range(0, image.shape[2]):
       new_dataset.write(image[:,:,ch], ch+1)
    else:
       new_dataset.write(image, 1)
    new_dataset.close()

    return True


histogan_ids = os.listdir('/home/mariapap/CODE/official/histogan/results_ReHistoGAN/reHistoGAN_model/')
for i, _ in enumerate(tqdm(histogan_ids)):
  id = histogan_ids[i]
  f = id.find('-gen')
  new_id = id[:f]
  hist_img = Image.open('/home/mariapap/CODE/official/histogan/results_ReHistoGAN/reHistoGAN_model/' + id)
  hist_img = np.array(hist_img)
#  print(hist_img)

  first_tif = rasterio.open('/home/mariapap/CODE/ChangeOS/REGIONS_June_01_15/{}.tif'.format(new_id))
  bounds = first_tif.bounds

  first_tif, hist_img = check_shape_and_resize(first_tif, hist_img)
#  print(first_tif.shape, hist_img.shape)

  save_tif_coregistered('{}/{}.tif'.format('/home/mariapap/CODE/ChangeOS/REGIONS_Dec23_HIST/',new_id), hist_img, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)

