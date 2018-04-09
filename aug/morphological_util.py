import glob
import os

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from imageio import imwrite
from skimage.transform import resize
from sklearn.cluster import KMeans
import skimage.morphology  # For using image labeling

from tqdm import tqdm


def overlay_masks(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    all_mask = []
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(image)
        overlayed_masks = np.sum(masks, axis=0)
        #target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        #os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        #imwrite(target_filepath, overlayed_masks)
        all_mask.append(overlayed_masks)

    return all_mask


def overlay_contours(images_dir, subdir_name, target_dir, touching_only=False):
    train_dir = os.path.join(images_dir, subdir_name)
    contours = []
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(get_contour(image))
        if touching_only:
            overlayed_masks = np.where(np.sum(masks, axis=0) > 128. + 255., 255., 0.).astype(np.uint8)
        else:
            overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        #target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        #os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        #imwrite(target_filepath, overlayed_masks)
        contours.append(overlayed_masks)

    return contours

def overlay_centers(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(get_center(image))
        overlayed_masks = np.where(np.sum(masks, axis=0) > 128., 255., 0.).astype(np.uint8)
        target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        imwrite(target_filepath, overlayed_masks)


def get_contour(img):
    img_contour = np.zeros_like(img).astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contour, contours, -1, (255, 255, 255), 4)
    return img_contour


def get_center(img):
    img_center = np.zeros_like(img).astype(np.uint8)
    y, x = ndi.measurements.center_of_mass(img)
    cv2.circle(img_center, (int(x), int(y)), 4, (255, 255, 255), -1)
    return img_center

def get_ground_truth(images_dir, subdir_name, target_dir):
    train_dir = os.path.join(images_dir, subdir_name)
    groud_truth = []
    for mask_dirname in tqdm(glob.glob('{}/*/masks'.format(train_dir))):
        masks = []
        for image_filepath in glob.glob('{}/*'.format(mask_dirname)):
            image = np.asarray(Image.open(image_filepath))
            image = image / 255.0
            masks.append(image)
        overlayed_masks = np.sum(masks, axis=0)
        #target_filepath = '/'.join(mask_dirname.replace(images_dir, target_dir).split('/')[:-1]) + '.png'
        #os.makedirs(os.path.dirname(target_filepath), exist_ok=True)
        #imwrite(target_filepath, overlayed_masks)

        lab_mask = skimage.morphology.label(overlayed_masks > 0.5)

        groud_truth.append(lab_mask)

    return groud_truth
