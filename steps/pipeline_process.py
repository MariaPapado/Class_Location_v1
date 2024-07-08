import torch
from PIL import Image
import numpy as np
import cv2
import os
import rasterio
import shutil
from shapely import geometry
from tqdm import tqdm
import subprocess


def calculate_intensity(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


ids = os.listdir('/home/mariapap/CODE/ChangeOS/REGIONS_Dec23/')
dir1 = '/home/mariapap/CODE/ChangeOS/REGIONS_Dec23/'
save_path = '/home/mariapap/CODE/ChangeOS/REGIONS_Dec23/'

for id in ids:
  img = Image.open('{}{}'.format(dir1, id))
  print(id)
  intns = calculate_intensity(img)
  if intns<=0.32:
    print('intns', intns)
    out = subprocess.call(["python", "thres_and_brighten_slide.py", "--img_path", "{}{}".format(dir1, id), "--save_path", "{}{}".format(save_path, id)])
    print('out', out)
