import time
import os
from options.test_options import TestOptions
from models.models import create_model
#from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import rasterio
from shapely import geometry


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.name = 'enlightening'
opt.model='single'
opt.which_direction='AtoB'
opt.which_model_netG='sid_unet_resize'
opt.skip=1
opt.use_norm = 1
opt.use_wgan = 0
opt.self_attention=True
opt.times_residual=True
opt.instance_norm=0
opt.resize_or_crop='no'
opt.which_epoch='200'
#opt.img_path = 'image.png'

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


def calculate_intensity(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


img_path = opt.img_path
bounds = rasterio.open(img_path).bounds
imgA = Image.open(img_path).convert('RGB')
img_intensity = calculate_intensity(imgA)
print('intensity', img_intensity)
#if img_intensity<0.28:
print('SHAPEEEEEEEEEEEE', np.array(imgA).shape)

imgA = np.array(imgA)
###############################################################
whole_pred = np.zeros((imgA.shape[0], imgA.shape[1], 3))
p=1000
s=900
#############################################################

if img_intensity<0.9:
    model = create_model(opt)

    transform_list = []
    transform_list += [transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]


    transform_list = transforms.Compose(transform_list)


#######################################################################################

    for x in range(0, whole_pred.shape[0], s):
        #print(image_before.tile_size)
        if x + p > whole_pred.shape[0]:
            x = whole_pred.shape[0] - p
        for y in range(0, whole_pred.shape[1], s):
            if y + p > whole_pred.shape[1]:
                y = whole_pred.shape[1] - p

            img = imgA[x:x+p, y:y+p, :]
            
#########################################################################################

            img = transform_list(img)
            #w = imgA.size(2)
            #h = imgA.size(1)

            input_img = img
            r,g,b = input_img[0]+1, input_img[1]+1, input_img[2]+1
            A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
            A_gray = torch.unsqueeze(A_gray, 0)

            img = img.unsqueeze(0)
            input_img = input_img.unsqueeze(0)
            A_gray = A_gray.unsqueeze(0)

            data = {'A': img, 'B': img, 'A_gray': A_gray, 'input_img': input_img, 'A_paths': img_path, 'B_paths': img_path}
            model.set_input(data)
            visuals = model.predict()
            img_path = model.get_image_paths()

            bright_img = visuals['fake_B']

            whole_pred[x:x+p, y:y+p, :] = bright_img                    


    whole_pred = np.array(whole_pred, dtype=np.uint8)
    save_tif_coregistered(opt.save_path, whole_pred, geometry.Polygon.from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top), channels = 3)
#    whole_pred = Image.fromarray(whole_pred)
#    whole_pred.save(opt.save_path)

else:
     print('Image does not need brightening. Intensity is above 0.28.')
