import torchvision.transforms.functional as F
from skimage import measure
import numpy as np
import torch
import cv2
from PIL import Image

class ChangeOS(object):
    def __init__(self, jit_model):
        self.model = jit_model
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(self.device)

    def __call__(self, pre_disaster_image, post_disaster_image):
        image = np.concatenate([pre_disaster_image, post_disaster_image], axis=2)
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        image = F.normalize(image,
                            mean=[123.675, 116.28, 103.53, 123.675, 116.28, 103.53],
#                            std=[58.395, 57.12, 65.03, 58.395, 57.12, 65.03],
                            std=[58.395, 57.12, 57.375, 58.395, 57.12, 57.375],
                            inplace=True)

        image = image.unsqueeze(0).to(self.device)
        loc, dam = self.model(image)
        #print(loc)

        loc, dam = object_based_infer(loc, dam) ##here loc is False True
        loc = loc.squeeze().astype(np.uint8)
        #print(np.unique(loc)) #np unique [0 1]
        dam = dam.squeeze().astype(np.uint8)

        return loc, dam

def object_based_infer(pre_logit, post_logit):
    loc = (pre_logit > 0.).cpu().squeeze(1).numpy()
    #print(loc)
    dam = post_logit.argmax(dim=1).cpu().squeeze(1).numpy()

    refined_dam = np.zeros_like(dam)
    for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
        refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

    return loc, refined_dam


def _object_vote(loc, dam):
    damage_cls_list = [1, 2, 3, 4]
    local_mask = loc
    #print(local_mask)
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, [8., 38., 25., 11.])]
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()
    return new_dam


def calculate_intensity(image):
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale


def histogram_match(source, reference, match_proportion=1.0):
    orig_shape = source.shape
    source = source.ravel()

    if np.ma.is_masked(reference):
        reference = reference.compressed()
    else:
        reference = reference.ravel()

    s_values, s_idx, s_counts = np.unique(
        source, return_inverse=True, return_counts=True)
    r_values, r_counts = np.unique(reference, return_counts=True)
    s_size = source.size

    if np.ma.is_masked(source):
        mask_index = np.ma.where(s_values.mask)
        s_size = np.ma.where(s_idx != mask_index[0])[0].size
        s_values = s_values.compressed()
        s_counts = np.delete(s_counts, mask_index)

    s_quantiles = np.cumsum(s_counts).astype(np.float64) / s_size
    r_quantiles = np.cumsum(r_counts).astype(np.float64) / reference.size

    interp_r_values = np.interp(s_quantiles, r_quantiles, r_values)

    if np.ma.is_masked(source):
        interp_r_values = np.insert(interp_r_values, mask_index[0], source.fill_value)

    target = interp_r_values[s_idx]

    if match_proportion is not None and match_proportion != 1:
        diff = source - target
        target = source - (diff * match_proportion)

    if np.ma.is_masked(source):
        target = np.ma.masked_where(s_idx == mask_index[0], target)
        target.fill_value = source.fill_value

    return target.reshape(orig_shape)


def multiple_preds(pre_disaster_image, post_disaster_image, model):
    loc0, dam = model(pre_disaster_image, post_disaster_image)
    loc1, dam = model(np.rot90(pre_disaster_image, k=1), np.rot90(post_disaster_image, k=1))
    loc2, dam = model(np.rot90(pre_disaster_image, k=2), np.rot90(post_disaster_image, k=2))
    loc3, dam = model(np.rot90(pre_disaster_image, k=3), np.rot90(post_disaster_image, k=3))
    loc4, dam = model(np.transpose(pre_disaster_image, (1,0,2)), np.transpose(post_disaster_image, (1,0,2)))

    loc =loc0 + np.rot90(loc1, k=3) + np.rot90(loc2, k=2) + np.rot90(loc3, k=1) + np.transpose(loc4, (1,0))
    idx1 = np.where(loc==5)
    idx0 = np.where(loc!=5)
    loc[idx0]=0
    loc[idx1]=1
    return loc


def make_prediction(img_before, img_after, model, p, s, id):
    whole_pred_before = np.zeros((img_before.shape[0], img_before.shape[1]))
    whole_pred_after = np.zeros((img_before.shape[0], img_before.shape[1]))

    for x in range(0, img_before.shape[0], s):
        #print(image_before.tile_size)
        if x + p > img_before.shape[0]:
            x = img_before.shape[0] - p
        for y in range(0, img_before.shape[1], s):
            if y + p > img_before.shape[1]:
                y = img_before.shape[1] - p
            img0 = img_before[x:x+p, y:y+p, :]
            img1 = img_after[x:x+p, y:y+p, :]
            h0, w0 = img0.shape[0], img0.shape[1]

            #img0 = cv2.resize(img0, (2*w0,2*h0), cv2.INTER_NEAREST)
            #img1 = cv2.resize(img1, (2*w0,2*h0), cv2.INTER_NEAREST)

            loc_before = multiple_preds(img0, img0, model)
            loc_after = multiple_preds(img1, img1, model)

            #loc_before = cv2.resize(loc_before, (w0,h0), cv2.INTER_NEAREST)
            #loc_after = cv2.resize(loc_after, (w0,h0), cv2.INTER_NEAREST)


    #        whole_pred[x:x+p, y:y+p] = whole_pred[x:x+p, y:y+p] + loc
            whole_pred_before[x:x+p, y:y+p] =  loc_before
            whole_pred_after[x:x+p, y:y+p] =  loc_after

    return whole_pred_before, whole_pred_after

