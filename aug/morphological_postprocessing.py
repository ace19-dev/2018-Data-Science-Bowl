import os

import ipywidgets as ipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib

masks, contours, ground_truth = joblib.load('../input/morphologicalpostprocessing/masks_contours_ground_truth_train.pkl')

def plot_list(images, labels):
    n_img = len(images)
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


for idx in range(5):
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    gt = ground_truth[idx]

    plot_list(images=[mask, contour], labels=[gt])



## clean masks
## Problem 1 dirty masks



idx = 5
dirty = masks[idx], contours[idx], ground_truth[idx]

plot_list(images = [dirty[0],dirty[1]],
          labels = [dirty[2]])



## Problem 2 dirty at border

idx = 44
dirty_at_border = masks[idx], contours[idx], ground_truth[idx]

plot_list(images = [dirty_at_border[0],dirty_at_border[1]],
          labels = [dirty_at_border[2]])



### Approach V1
### Let's build a function that calculates the average size of the nuclei.
### We will need to to choose the structural element for our postprocessing

import skimage.morphology as morph
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from scipy.stats import itemfreq

def mean_blob_size(mask):
    labels, labels_nr = ndi.label(mask)
    if labels_nr < 2:
        mean_area = 1
        mean_radius = 1
    else:
        mean_area = int(itemfreq(labels)[1:, 1].mean())
        mean_radius = int(np.round(np.sqrt(mean_area) / np.pi))
    return mean_area, mean_radius

m, c, t = dirty

m_b = m > threshold_otsu(m)
c_b = c > threshold_otsu(c)

plot_list(images=[m_b,c_b],labels=[])

m_ = np.where(m_b | c_b, 1, 0)
plot_list(images=[m_],labels=[])

m_ = ndi.binary_fill_holes(m_)
plot_list(images=[m_],labels=[])

m_ = np.where(c_b & (~m_b), 0, m_)
plot_list(images=[m_],labels=[])

area, radius = mean_blob_size(m_b)
m_ = morph.binary_opening(m_, selem=morph.disk(0.25*radius))
plot_list(images=[m_],labels=[])


def clean_mask_v1(m, c):
    m_b = m > threshold_otsu(m)
    c_b = c > threshold_otsu(c)

    m_ = np.where(m_b | c_b, 1, 0)
    m_ = ndi.binary_fill_holes(m_)
    m_ = np.where(c_b & (~m_b), 0, m_)

    area, radius = mean_blob_size(m_b)
    m_ = morph.binary_opening(m_, selem=morph.disk(0.25 * radius))
    return m_

m, c, t = dirty

m_ = clean_mask_v1(m,c)

plot_list(images = [m,c,m_],
          labels = [t]
         )

m,c,t = dirty_at_border
m_ = clean_mask_v1(m,c)

plot_list(images = [m,c,m_],
          labels = [t]
         )


### Approach V2¶
### Let's start by binarizing and filling the holes again

m,c,t = dirty_at_border

# threshold
m_b = m > threshold_otsu(m)
c_b = c > threshold_otsu(c)

# combine contours and masks and fill the cells
m_ = np.where(m_b | c_b, 1, 0)
m_ = ndi.binary_fill_holes(m_)
plot_list(images=[m_],labels=[])

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

# close what wasn't closed before
area, radius = mean_blob_size(m_b)
struct_size = int(1.25*radius)
struct_el = morph.disk(struct_size)
m_padded = pad_mask(m_, pad=struct_size)
m_padded = morph.binary_closing(m_padded, selem=struct_el)
m_ = crop_mask(m_padded, crop=struct_size)
plot_list(images=[m_],labels=[])

m_ = np.where(c_b & (~m_b), 0, m_)
plot_list(images=[m_],labels=[])

area, radius = mean_blob_size(m_b)
struct_size = int(0.75*radius)
struct_el = morph.disk(struct_size)
m_padded = pad_mask(m_, pad=struct_size)
m_padded = morph.binary_opening(m_padded, selem=struct_el)
m_ = crop_mask(m_padded, crop=struct_size)
plot_list(images=[m_,m],labels=[t])

# join the connected cells with what we had at the beginning
m_ = np.where(m_b|m_,1,0)
m_ = ndi.binary_fill_holes(m_)
plot_list(images=[m_,m],labels=[t])

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

m_ = drop_artifacts(m_, m_b,min_coverage=0.25)
plot_list(images=[m_,m,c],labels=[t])


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

m,c,t = dirty_at_border
m_ = clean_mask_v2(m,c)

plot_list(images = [m,c,m_],
          labels = [t]
         )

for idx in range(5):
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    gt = ground_truth[idx]

    plot_list(images=[mask, contour, cleaned_mask], labels=[gt])



## Problem 3 not everything gets filled

for idx in [38]:
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    gt = ground_truth[idx]

    plot_list(images=[mask, contour, cleaned_mask], labels=[gt])


## Problem 4 some cells get connected

for idx in [0]:
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    gt = ground_truth[idx]

    plot_list(images=[mask, contour, cleaned_mask], labels=[gt])


## Approach v1
## In this approach we will simply cut the masks with the contours and use that as markers.
## Simple but really effective.

def good_markers_v1(m, c):
    # threshold
    m_b = m > threshold_otsu(m)
    c_b = c > threshold_otsu(c)

    mk_ = np.where(c_b, 0, m)
    return mk_


for idx in range(5):
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v1(mask, contour)
    gt = ground_truth[idx]

    plot_list(images=[mask, contour, cleaned_mask, good_markers], labels=[gt])




## Problem 1 building markers on initial mask when we have better mask
## There is no point in using initial masks when we worked so hard on making them better right?
## Let's use our results from the first step

def good_markers_v2(m_b, c):
    # threshold
    c_thresh = threshold_otsu(c)
    c_b = c > c_thresh

    mk_ = np.where(c_b, 0, m_b)
    return mk_


for idx in range(5):
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v2(cleaned_mask, contour)
    gt = ground_truth[idx]

    plot_list(images=[mask, contour, cleaned_mask, good_markers], labels=[gt])



## Problem 2 some markers are to large and connected
## Unfortunately it is not perfect. If we have to connected cells and we have one connected marker
## for those cells watershed will not detach it. We need to make them better,
## smaller and positioned more in the center of nuceli.

m,c,t = dirty

cleaned_mask = clean_mask_v2(m, c)

c_b = c > threshold_otsu(c)
mk_ = np.where(c_b,0,cleaned_mask)
plot_list(images=[m,c,mk_],labels=[])

area, radius = mean_blob_size(m_b)
struct_size = int(0.25*radius)
struct_el = morph.disk(struct_size)
m_padded = pad_mask(mk_, pad=struct_size)
m_padded = morph.erosion(m_padded, selem=struct_el)
mk_ = crop_mask(m_padded, crop=struct_size)
plot_list(images=[m,c,mk_],labels=[])

mk_,_ = ndi.label(mk_)
plot_list(images=[cleaned_mask],labels=[mk_, t])


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


for idx in range(5):
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v3(cleaned_mask, contour)
    gt = ground_truth[idx]

    plot_list(images=[mask, contour, cleaned_mask], labels=[good_markers, gt])



## Problem 3 still some barely connected markers are left
## Unfortunately for some images the markers are not eroded enough and are left connected (look at the orange blob at the bottom right corner in the forth column).
## Some tweaking should improve it but beware that for other images it might decrease the score.

for idx in [25, 27]:
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v3(cleaned_mask, contour)
    gt = ground_truth[idx]

    plot_list(images=[mask, contour, cleaned_mask], labels=[good_markers, gt])



## Problem 4 we are dropping markers on many images with small cells
## Good distance

def good_markers_v4(m_b,c):
    return

def good_distance_v1(m_b):
    distance = ndi.distance_transform_edt(m_b)
    return distance


for idx in range(5):
    print(idx)
    mask = masks[idx]
    contour = contours[idx]
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v3(cleaned_mask, contour)
    good_distance = good_distance_v1(cleaned_mask)
    gt = ground_truth[idx]

    plot_list(images=[cleaned_mask, good_distance], labels=[good_markers, gt])




## Watershed

def watershed_v1(mask, contour):
    cleaned_mask = clean_mask_v2(mask, contour)
    good_markers = good_markers_v3(cleaned_mask, contour)
    good_distance = good_distance_v1(cleaned_mask)

    water = morph.watershed(-good_distance, good_markers, mask=cleaned_mask)

    return water


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








from itertools import product

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


test_masks = joblib.load('../input/test-predictions/test_masks.pkl')

from skimage.color import label2rgb

for mask in test_masks[:5]:
    plt.imshow(label2rgb(mask))
    plt.show()
