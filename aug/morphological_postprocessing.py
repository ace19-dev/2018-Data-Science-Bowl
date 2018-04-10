import glob
import os
import sys
import argparse
from tqdm import tqdm

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib
import cv2
import skimage.morphology as morph
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from scipy.stats import itemfreq
from PIL import Image

from aug.morphological_util import get_ground_truth, overlay_contours, overlay_masks

import tensorflow as tf


FLAGS = None



def plot_list(images, labels):

    n_img = len(images)
    '''
    n_lab = len(labels)
    n = n_lab+n_img
    plt.figure(figsize=(12,8))
    for i, image in enumerate(images):
        plt.subplot(1,n,i+1)
        plt.imshow(image)
    for j, label in enumerate(labels):
        plt.subplot(1,n,n_img+j+1)
        plt.imshow(label, cmap='nipy_spectral')
    plt.show()
    '''

# calculates the average size of the nuclei.
# We will need to to choose the structural element for our postprocessing
def mean_blob_size(mask):
    labels, labels_nr = ndi.label(mask)
    if labels_nr < 2:
        mean_area = 1
        mean_radius = 1
    else:
        mean_area = int(itemfreq(labels)[1:, 1].mean())
        mean_radius = int(np.round(np.sqrt(mean_area) / np.pi))
    return mean_area, mean_radius


def clean_mask_v1(m, c):
    m_b = m > threshold_otsu(m)
    c_b = c > threshold_otsu(c)

    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)
    m_ = np.where(c_b & (~m_b), 0, m_)

    area, radius = mean_blob_size(m_b)
    m_ = morph.binary_opening(m_, selem=morph.disk(0.25 * radius))
    return m_


def pad_mask(mask, pad):
    if pad <= 1:
        pad = 2
    h, w = mask.shape
    h_pad = h + 2 * pad
    w_pad = w + 2 * pad
    mask_padded = np.zeros((h_pad, w_pad))
    mask_padded[pad:pad + h, pad:pad + w] = mask
    mask_padded[pad - 1, :] = 1
    mask_padded[pad + h + 1, :] = 1
    mask_padded[:, pad - 1] = 1
    mask_padded[:, pad + w + 1] = 1

    return mask_padded


def crop_mask(mask, crop):
    if crop <= 1:
        crop = 2
    h, w = mask.shape
    mask_cropped = mask[crop:h - crop, crop:w - crop]
    return mask_cropped


def drop_artifacts(mask_after, mask_pre, min_coverage=0.5):
    connected, nr_connected = ndi.label(mask_after)
    mask = np.zeros_like(mask_after)
    for i in range(1, nr_connected + 1):
        conn_blob = np.where(connected == i, 1, 0)
        initial_space = np.where(connected == i, mask_pre, 0)
        blob_size = np.sum(conn_blob)
        initial_blob_size = np.sum(initial_space)
        coverage = float(initial_blob_size) / float(blob_size)
        if coverage > min_coverage:
            mask = mask + conn_blob
        else:
            mask = mask + initial_space
    return mask


def clean_mask_v2(m, c):
    # threshold
    m_b = m > threshold_otsu(m)
    c_b = c > threshold_otsu(c)

    # combine contours and masks and fill the cells
    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)

    # close what wasn't closed before
    area, radius = mean_blob_size(m_b)
    struct_size = int(1.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(m_, pad=struct_size)
    m_padded = morph.binary_closing(m_padded, selem=struct_el)
    m_ = crop_mask(m_padded, crop=struct_size)

    # open to cut the real cells from the artifacts
    area, radius = mean_blob_size(m_b)
    struct_size = int(0.75 * radius)
    struct_el = morph.disk(struct_size)
    m_ = np.where(c_b & (~m_b), 0, m_)
    m_padded = pad_mask(m_, pad=struct_size)
    m_padded = morph.binary_opening(m_padded, selem=struct_el)
    m_ = crop_mask(m_padded, crop=struct_size)

    # join the connected cells with what we had at the beginning
    m_ = np.where(m_b | m_, 1, 0)
    m_ = ndi.binary_fill_holes(m_)

    # drop all the cells that weren't present at least in 25% of area in the initial mask
    m_ = drop_artifacts(m_, m_b, min_coverage=0.25)

    return m_


def good_markers_v1(m, c):
    # threshold
    m_b = m > threshold_otsu(m)
    c_b = c > threshold_otsu(c)

    mk_ = np.where(c_b, 0, m)
    return mk_


def good_markers_v2(m_b, c):
    # threshold
    c_thresh = threshold_otsu(c)
    c_b = c > c_thresh

    mk_ = np.where(c_b, 0, m_b)
    return mk_


def good_markers_v3(m_b, c):
    # threshold
    c_b = c > threshold_otsu(c)

    mk_ = np.where(c_b, 0, m_b)

    area, radius = mean_blob_size(m_b)
    struct_size = int(0.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(mk_, pad=struct_size)
    m_padded = morph.erosion(m_padded, selem=struct_el)
    mk_ = crop_mask(m_padded, crop=struct_size)
    mk_, _ = ndi.label(mk_)
    return mk_


## Problem 4 we are dropping markers on many images with small cells
## Good distance
def good_distance_v1(m_b):
    distance = ndi.distance_transform_edt(m_b)
    return distance


def watershed_v1(mask, contour):
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v3(cleaned_mask, contour)
    good_distance = good_distance_v1(cleaned_mask)

    water = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

    return water


def add_dropped_water_blobs(water, mask_cleaned):
    water_mask = (water > 0).astype(np.uint8)
    dropped = mask_cleaned - water_mask
    dropped, _ = ndi.label(dropped)
    dropped = np.where(dropped, dropped + water.max(), 0)
    water = water + dropped
    return water


def drop_artifacts_per_label(labels, initial_mask):
    labels_cleaned = np.zeros_like(labels)
    for i in range(1, labels.max() + 1):
        component = np.where(labels == i, 1, 0)
        component_initial_mask = np.where(labels == i, initial_mask, 0)
        component = drop_artifacts(component, component_initial_mask)
        labels_cleaned = labels_cleaned + component * i
    return labels_cleaned


def watershed_v2(mask, contour):
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v3(cleaned_mask, contour)
    good_distance = good_distance_v1(cleaned_mask)

    water = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

    water = add_dropped_water_blobs(water, cleaned_mask)

    m_thresh = threshold_otsu(mask)
    initial_mask_binary = (mask > m_thresh).astype(np.uint8)
    water = drop_artifacts_per_label(water, initial_mask_binary)
    return water


def relabel(img):
    h, w = img.shape

    relabel_dict = {}

    for i, k in enumerate(np.unique(img)):
        if k == 0:
            relabel_dict[k] = 0
        else:
            relabel_dict[k] = i
    for i, j in product(range(h), range(w)):
        img[i, j] = relabel_dict[img[i, j]]
    return img


def drop_small(img, min_size):
    img = morph.remove_small_objects(img, min_size=min_size)
    return relabel(img)


def fill_holes_per_blob(image):
    image_cleaned = np.zeros_like(image)
    for i in range(1, image.max() + 1):
        mask = np.where(image == i, 1, 0)
        mask = ndi.morphology.binary_fill_holes(mask)
        image_cleaned = image_cleaned + mask * i
    return image_cleaned


def watershed_v3(mask, contour):
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v3(cleaned_mask, contour)
    good_distance = good_distance_v1(cleaned_mask)

    labels = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

    labels = add_dropped_water_blobs(labels, cleaned_mask)

    m_thresh = threshold_otsu(mask)
    initial_mask_binary = (mask > m_thresh).astype(np.uint8)
    labels = drop_artifacts_per_label(labels, initial_mask_binary)

    labels = drop_small(labels, min_size=20)
    labels = fill_holes_per_blob(labels)

    return labels



def main(_):
    ground_truth = get_ground_truth(images_dir=FLAGS.images_dir,
                                    subdir_name=FLAGS.subdir_name,
                                    target_dir=None)

    contours = overlay_contours(images_dir=FLAGS.images_dir,
                                subdir_name=FLAGS.subdir_name,
                                target_dir=None)

    masks = overlay_masks(images_dir=FLAGS.images_dir,
                          subdir_name=FLAGS.subdir_name,
                          target_dir=None)


    ############################
    # Problem 1 -> dirty masks
    ############################
    idx = 5
    dirty = masks[idx], contours[idx], ground_truth[idx]
    plot_list(images=[dirty[0], dirty[1]],
              labels=[dirty[2]])

    #################################
    # Problem 2 -> dirty at border
    #################################
    idx = 44
    dirty_at_border = masks[idx], contours[idx], ground_truth[idx]
    plot_list(images=[dirty_at_border[0], dirty_at_border[1]],
              labels=[dirty_at_border[2]])

    ################
    # Approach V1
    ################
    m, c, t = dirty

    ########################################################################
    #  Let's put it all together in a function - def clean_mask_v1(m,c)
    #
    #  m, c, t = dirty
    #  m_ = clean_mask_v1(m,c)
    #  plot_list(images = [m,c,m_], labels = [t])
    #
    ########################################################################
    # let's proceed to cleaning.
    # First we binarize both the mask and contours using global, otsu thresholding method:
    m_b = m > threshold_otsu(m)
    c_b = c > threshold_otsu(c)
    plot_list(images=[m_b, c_b], labels=[])

    # combine binarized masks and contours
    m_ = np.where(m_b | c_b, 1, 0)
    plot_list(images=[m_], labels=[])

    # fill the holes that remained
    m_ = ndi.binary_fill_holes(m_)
    plot_list(images=[m_], labels=[])

    # Now that we filled the holes in the cells we can detach them again because we have the contour information
    m_ = np.where(c_b & (~m_b), 0, m_)
    plot_list(images=[m_], labels=[])

    # We are left with artifacts. Let's use binary_openning to drop them.
    area, radius = mean_blob_size(m_b)
    m_ = morph.binary_opening(m_, selem=morph.disk(0.25 * radius))
    plot_list(images=[m_], labels=[])



    # It works to a certain extend but it removes things that
    # where not connected and/or things around borders

    ################
    # Approach V2
    ################
    # Let's start by binarizing and filling the holes again
    m, c, t = dirty_at_border

    ########################################################################
    #  Let's put it all together in one function - def clean_mask_v2(m,c)
    #
    #  m,c,t = dirty_at_border
    #  m_ = clean_mask_v2(m,c)
    #
    #  plot_list(images = [m,c,m_], labels = [t])
    #
    ########################################################################

    # threshold
    m_b = m > threshold_otsu(m)
    c_b = c > threshold_otsu(c)

    # combine contours and masks and fill the cells
    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)
    plot_list(images=[m_], labels=[])


    # Now we will use binary_closing to fill what wasn't closed with fill holes.
    # We will need two helper functions pad_mask and crop_mask to deal with problems around the edges

    # close what wasn't closed before
    area, radius = mean_blob_size(m_b)
    struct_size = int(1.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(m_, pad=struct_size)
    m_padded = morph.binary_closing(m_padded, selem=struct_el)
    m_ = crop_mask(m_padded, crop=struct_size)
    plot_list(images=[m_], labels=[])

    # we closed everything but it is way more than we wanted.
    # Let's now cut it with our contours and see what we get
    m_ = np.where(c_b & (~m_b), 0, m_)
    plot_list(images=[m_], labels=[])

    # we can use binary_openning with a larger structural element. Let's try that
    area, radius = mean_blob_size(m_b)
    struct_size = int(0.75 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(m_, pad=struct_size)
    m_padded = morph.binary_opening(m_padded, selem=struct_el)
    m_ = crop_mask(m_padded, crop=struct_size)
    plot_list(images=[m_, m], labels=[t])

    # join the connected cells with what we had at the beginning
    m_ = np.where(m_b | m_, 1, 0)
    m_ = ndi.binary_fill_holes(m_)
    plot_list(images=[m_, m], labels=[t])

    m_ = drop_artifacts(m_, m_b, min_coverage=0.25)
    plot_list(images=[m_, m, c], labels=[t])


    ############################################
    # Problem 3 -> not everything gets filled
    ############################################

    ############################################
    # Problem 4 -> some cells get connected
    #
    # Ideas:
    # - work more with dilation
    # - do better around borders
    # - drop some cells after watershed with drop_artifact function
    #
    # TODO: clean_mask_V3 would be dev...
    # Go ahead and try to improve it. The floor is yours
    #
    # def clean_mask_v3(m,c):
    #     return
    #
    ############################################


    ###################
    # Good Markers
    ###################
    # In this approach we will simply cut the masks with the contours and use that as markers.
    # Simple but really effective.
    for idx in range(5):
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v1(mask, contour)
        gt = ground_truth[idx]

        plot_list(images=[mask, contour, cleaned_mask, good_markers], labels=[gt])

    # Problem 1 -> building markers on initial mask when we have better mask
    for idx in range(5):
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v2(cleaned_mask, contour)
        gt = ground_truth[idx]

        plot_list(images=[mask, contour, cleaned_mask, good_markers], labels=[gt])

    # Problem 2 some markers are to large and connected
    m, c, t = dirty
    cleaned_mask = clean_mask_v2(m, c)
    c_b = c > threshold_otsu(c)
    mk_ = np.where(c_b, 0, cleaned_mask)
    plot_list(images=[m, c, mk_], labels=[])

    # For instance the two markers at the top left are still connected and will be treated
    # as a single marker by the watershed And nowe lets erode the markers
    area, radius = mean_blob_size(m_b)
    struct_size = int(0.25 * radius)
    struct_el = morph.disk(struct_size)
    m_padded = pad_mask(mk_, pad=struct_size)
    m_padded = morph.erosion(m_padded, selem=struct_el)
    mk_ = crop_mask(m_padded, crop=struct_size)
    plot_list(images=[m, c, mk_], labels=[])

    # we now compare those markers with the labels we get the following
    mk_, _ = ndi.label(mk_)
    plot_list(images=[cleaned_mask], labels=[mk_, t])


    #########################################################
    # So the markers and cleaned mask look really good!
    #########################################################
    for idx in range(5):
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v3(cleaned_mask, contour)
        gt = ground_truth[idx]

        plot_list(images=[mask, contour, cleaned_mask], labels=[good_markers, gt])



    # Problem 3 -> still some barely connected markers are left¶
    for idx in [25, 27]:
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v3(cleaned_mask, contour)
        gt = ground_truth[idx]

        plot_list(images=[mask, contour, cleaned_mask], labels=[good_markers, gt])

    #########################################################################
    # Problem 4 -> we are dropping markers on many images with small cells
    #
    # Ideas
    # - play with binary closing/opening
    # - involve contours and/or centers in this
    # - we will asume that lost markers are in facet small cells that don't need to be divided and
    #   we will get back all the cells that were dropped in watershed
    # - use local maxima on distance transform
    #
    # TODO: good_markers_v4 need to be dev...
    # def good_markers_v4(m_b,c):
    #     return
    #
    #########################################################################

    #####################
    # Good distance
    #####################
    # Here I have no better idea than to use the binary distance from the background.
    # Feel free to improve on that!
    #
    # Idea
    # - investigate imposing some gradients on original image or good clean mask
    #
    for idx in range(5):
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v3(cleaned_mask, contour)
        good_distance = good_distance_v1(cleaned_mask)
        gt = ground_truth[idx]

        plot_list(images=[cleaned_mask, good_distance], labels=[good_markers, gt])


    ########################
    # Watershed
    ########################
    for idx in range(5):
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v3(cleaned_mask, contour)
        good_distance = good_distance_v1(cleaned_mask)

        water = watershed_v1(mask, contour)

        gt = ground_truth[idx]

        plot_list(images=[cleaned_mask, good_distance], labels=[good_markers, water, gt])


    # Problem 1 -> some cells are dumped

    # Problem 2 -> some artifacts from mask_cleaning remain
    # Unfortunatelly some cells are dropped, some cells are oversegmented and
    # some artifacts from the mask cleaning still remain.
    # The good thing is we can deal with some of those problems by using ideas we have already tried.
    for idx in range(5):
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v3(cleaned_mask, contour)
        good_distance = good_distance_v1(cleaned_mask)

        water = watershed_v2(mask, contour)

        gt = ground_truth[idx]

        plot_list(images=[cleaned_mask, good_distance], labels=[good_markers, water, gt])


    # Problem 3 -> some cells are oversemgmented and small cell chunks remain
    # Now some small pieces of cells may remain at this point or the cells could get oversegmented.
    # We will deal with that by dropping to small to be a cell blobs.
    for idx in range(5):
        print(idx)
        mask = masks[idx]
        contour = contours[idx]
        cleaned_mask = clean_mask_v2(mask, contour)
        good_markers = good_markers_v3(cleaned_mask, contour)
        good_distance = good_distance_v1(cleaned_mask)

        water = watershed_v3(mask, contour)

        gt = ground_truth[idx]

        plot_list(images=[cleaned_mask, good_distance], labels=[good_markers, water, gt])

    idx = 0
    train_dir = os.path.join(FLAGS.images_dir, FLAGS.subdir_name)
    for filename in tqdm(glob.glob('{}/*'.format(train_dir))):

        imagename = filename.split("/")[-1]

        mask = masks[idx]
        contour = contours[idx]

        water = watershed_v3(mask, contour)
        img = Image.fromarray(water.astype('uint8'))
        water_path = (filename + '/water/')
        if not os.path.exists(water_path):
            os.makedirs(water_path)
        img.save(os.path.join(water_path, imagename + '.png'))

        '''
        cleaned_mask = clean_mask_v2(mask, contour)
        img = Image.fromarray(cleaned_mask.astype('uint8'))
        cleaned_mask_path = (filename + '/cleaned_mask/')
        if not os.path.exists(cleaned_mask_path):
            os.makedirs(cleaned_mask_path)
        img.save(os.path.join(cleaned_mask_path, imagename + '.png'))

        good_markers = good_markers_v3(cleaned_mask, contour)
        img = Image.fromarray(good_markers)
        good_markers_path = (filename + '/good_markers/')
        if not os.path.exists(good_markers_path):
            os.makedirs(good_markers_path)
        img.save(os.path.join(good_markers_path, imagename + '.png'))

        good_distance = good_distance_v1(cleaned_mask)
        img = Image.fromarray(good_distance.astype('uint8'))
        good_distance_path = (filename + '/good_distance/')
        if not os.path.exists(good_distance_path):
            os.makedirs(good_distance_path)
        img.save(os.path.join(good_distance_path, imagename + '.png'))
        '''
        idx = idx + 1






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
         '--images_dir',
         default='../../../dl_data/nucleus',
         type=str,
         help="Image directory")

    parser.add_argument(
         '--subdir_name',
         default='stage1_train',
         type=str,
         help="Sub directory name")

    #parser.add_argument(
    #     '--target_dir',
    #     default='stage1_train',
    #     type=str,
    #     help="Sub directory name")

    parser.add_argument(
        '--img_size',
        type=int,
        default=256,
        help="Image height and width")

    parser.add_argument(
        '--aug_prefix',
        default='_aug_',
        type=str,
        help="prefix name of augmentation")

    parser.add_argument(
        '--aug_count',
        type=int,
        default=2,
        help="Count of augmentation")

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
