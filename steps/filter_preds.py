import numpy as np
import cv2 
from skimage import morphology
import os
import rasterio
from shapely import geometry
from tqdm import tqdm

def check_pinks(after_img_path):
    lower = (120,75,130)
    upper = (180,130,350)

    rgb_im = cv2.imread(after_img_path, cv2.IMREAD_UNCHANGED)
    hsv = cv2.cvtColor(rgb_im, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    result = cv2.bitwise_and(rgb_im, rgb_im, mask=mask)

    nonblack = np.all(np.not_equal(result, [0,0,0]), axis=-1)
    idx_nonblack = np.where(nonblack==True)

    #cv2.imwrite('out_pink.png', result)
    return nonblack, idx_nonblack


def filter_pinks(nonblack, output_pred_path):
    im = cv2.imread(output_pred_path, cv2.IMREAD_UNCHANGED)

    tree_bin = np.zeros((im.shape[0], im.shape[1]))
    idx_tree_bin = np.where(nonblack==True)
    tree_bin[idx_tree_bin] =4 #unique 0, 4

    label_bin = np.zeros((im.shape[0], im.shape[1]))
    reds = np.all(np.equal(np.array(im, dtype=np.uint8), [0,255,0]), axis=-1)
    greens = np.all(np.equal(np.array(im, dtype=np.uint8), [0,0,255]), axis=-1)
    label_bin[reds==True] = 1
    label_bin[greens==True] = 1

    contours, hierarchy = cv2.findContours(np.array(label_bin, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
      if len(cnt)>2:
        im_draw = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
        cv2.fillPoly(im_draw, [cnt], color=(255,255,255))
        idx = np.where(im_draw[:,:,0]==255)
        tree_values = tree_bin[idx]
        tree_idx4 = np.where(tree_values==4)
        if len(tree_idx4[0])>5:
          cv2.fillPoly(im, [cnt], color=(0,0,0))

    return im


def check_contours(im, bw):
    contours, hierarchy = cv2.findContours(np.array(bw, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    del_contours = []
    if contours:
      for cnt in contours:
        if cv2.contourArea(cnt)<=15 or len(cnt)<=2:
          del_contours.append(cnt)

        if del_contours:
          cv2.fillPoly(im, del_contours, color=(0,0,0))

    return im


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

after_imgs_dir = '/home/mariapap/CODE/ChangeOS/REGIONS_Dec/'
outputs_dir = '/home/mariapap/CODE/ChangeOS/PREP_WHOLE_PIPELINE/PIPELINE_RESULTS/OUTPUT/'
ids = os.listdir(after_imgs_dir)

for _, id in tqdm(enumerate(ids)):
    print(id)
    #im = cv2.imread('output_9414115_834.tif', cv2.IMREAD_UNCHANGED)
    im = rasterio.open(outputs_dir + 'output_{}'.format(id))
    bounds = im.bounds
    im = im.read()
    nonblack, idx_nonblack = check_pinks(after_imgs_dir + id)

    if len(idx_nonblack[0])!=0:
      im = filter_pinks(nonblack, outputs_dir + 'output_{}'.format(id))


    reds = np.all(np.equal(im, [0,255,0]), axis=-1)
    if len(reds[0]!=0):
      bw = np.zeros((im.shape[0], im.shape[1]))
      bw[reds==True] = 1
      im = check_contours(im, bw)


    reds = np.all(np.equal(im, [0,0,255]), axis=-1)
    if len(reds[0]!=0):
      bw = np.zeros((im.shape[0], im.shape[1]))
      bw[reds==True] = 1
      im = check_contours(im, bw)


    save_tif_coregistered('./outputs/' + 'output_{}'.format(id), im[:,:,[2,1,0]], geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)


################################################################################################################
