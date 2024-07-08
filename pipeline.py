import torch
from PIL import Image
import numpy as np
import cv2
import os
import rasterio
import shutil
from tools import *
from shapely import geometry
from buildings import *
from tqdm import tqdm

dir_before = '/home/mariapap/CODE/ChangeOS/REGIONS_August/'
dir_after = '/home/mariapap/CODE/ChangeOS/REGIONS_Dec/'
#dir_after = '/home/mariapap/CODE/official/histogan/class_location_outputs_rawhist/'

#dir_after = '/home/mariapap/CODE/HistoGAN/results_ReHistoGAN/reHistoGAN_model/class_location_outputs/'
#dir_after = '/home/mariapap/CODE/official/histogan/class_location_outputs/'
#dir_after = '/home/mariapap/CODE/lcdpnet/class_location_outputs/'
#dir_after = '/home/mariapap/CODE/official2/enlightengan/class_location_output/'
#dir_after = '/home/mariapap/CODE/official/histogan/class_location_outputs_enlightengan/'

ids = os.listdir('/home/mariapap/CODE/ChangeOS/REGIONS_Dec/')
ids_August = os.listdir('/home/mariapap/CODE/ChangeOS/REGIONS_August/')

####folders
f_pipeline_result = './PIPELINE_RESULTS'
#f_before = f_pipeline_result + '/BEFORE'
#f_after = f_pipeline_result + '/AFTER'
f_after_registered = f_pipeline_result + '/AFTER_REGISTERED'
f_pred_before = f_pipeline_result + '/PREDS_BEFORE'
f_pred_after = f_pipeline_result + '/PREDS_AFTER'
f_output = f_pipeline_result + '/OUTPUT'
#f_output_disappear = f_pipeline_result + '/OUTPUT_DISAPPEAR'
##############


if os.path.exists(f_pipeline_result):
    shutil.rmtree(f_pipeline_result)
os.mkdir(f_pipeline_result)

#os.mkdir(f_before)
#os.mkdir(f_after)
os.mkdir(f_after_registered)
os.mkdir(f_pred_before)
os.mkdir(f_pred_after)
os.mkdir(f_output)


#ids = ['9316019_127.tif']
#ids = ['9352495_399.tif']
#ids = ['9414115_834.tif']

for _, id in enumerate(tqdm(ids)):
  if id in ids_August:
    print(id)

    bef_img_cv2 = cv2.imread(dir_before + id, cv2.IMREAD_UNCHANGED)
    aft_img_cv2 = cv2.imread(dir_after + id, cv2.IMREAD_UNCHANGED)

    bef_intns = calculate_intensity(Image.fromarray(bef_img_cv2))
    aft_intns = calculate_intensity(Image.fromarray(aft_img_cv2))  

    #print('intns', bef_intns, aft_intns)

    raster_before = rasterio.open(dir_before + id)
    raster_after = rasterio.open(dir_after +  id)

    ####1. resize images if different shapes
    im_before, im_after, bounds = check_shape_and_resize(raster_before, raster_after)

    #print(im_before.bounds)
#    save_tif_coregistered('{}/before_{}'.format(f_before,id), im_before, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)
#    save_tif_coregistered('{}/after_{}'.format(f_after,id), im_after, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)

    #print(im_before.shape, im_after.shape)
    ###2. register image pair

    print(im_before.shape, im_after.shape)


    #blanks_before = np.where(im_before[:,:,0]<50)
    #blanks_after = np.where(im_after[:,:,0]<50)

    #area = im_before.shape[0]*im_after.shape[1]

    #area_blank_before = len(blanks_before[0])
    #area_blank_after = len(blanks_after[0])

    #perc_before = area_blank_before/area
    #perc_after = area_blank_after/area

    #print(perc_before, perc_after)


    #if perc_before<0.5 and perc_after<0.5:
    #if np.abs(aft_intns-bef_intns) > 0.10:

          #print('tadaaaaaaaaaa')

    #      if bef_intns > aft_intns:
    #          print('bef')
              #histogram_match(source, ref)
    #          im_before = histogram_match(bef_img_cv2, aft_img_cv2)
    #          cv2.imwrite('img_before_{}'.format(id), np.array(im_before, dtype=np.uint8))
    #      else:
    #          print('aft')
              #print(img_before, img_after)
   #           im_after = histogram_match(aft_img_cv2, bef_img_cv2)    
   #           cv2.imwrite('img_after_{}'.format(id), np.array(im_after, dtype=np.uint8)) 

    try:
          im_after_transformed = register_image_pair(im_before, im_after)
    except:
          im_after_transformed = im_after
    save_tif_coregistered('{}/after_transformed_{}'.format(f_after_registered,id), im_after_transformed, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)
    #else:
    #    im_after_transformed = im_after
    #    print('NOPE!')

    ###3. find before and after building masks
    model = torch.jit.load('/home/mariapap/CODE/ChangeOS/models/changeos_r101.pt')
    model.eval()
    model = ChangeOS(model)

    patch_size, step = 512, 512
    buildings_before, buildings_after = make_prediction(im_before, im_after_transformed, model, patch_size, step, id)
    save_tif_coregistered('{}/{}'.format(f_pred_before,id), buildings_before*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)
    save_tif_coregistered('{}/{}'.format(f_pred_after,id), buildings_after*255, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 1)

    #output = post_processing(buildings_before, buildings_after)
    #output_disappears = post_processing(buildings_after, buildings_before)

    output_appears = apply_watershed(buildings_before, buildings_after)
    output_disappears = apply_watershed(buildings_after, buildings_before)
    #print('outtttttt', output_appears.shape)
    #print('before', np.unique(output_appears))
    #print('after', np.unique(output_disappears))
    idx2 = np.where(output_disappears==1)
    output_appears[idx2]=2

    output = visualize(output_appears)

    save_tif_coregistered('{}/output_{}'.format(f_output,id), output, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)












    
