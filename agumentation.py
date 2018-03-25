import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import cv2

import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

print('Dependencies Loaded')
# Set some parameters
IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3
TRAIN_PATH = '../../dl_data/nuclei_dataset/stage1_train/'
TEST_PATH = '../../dl_data/nuclei_dataset/stage1_test/'

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
seed = 42
random.seed = seed
np.random.seed = seed

# Get train and test IDs
train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

# Get and resize train images and masks
X_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_train[n] = img
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)
    Y_train[n] = mask

# Get and resize test images
X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:, :, :IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img

print('Done!')

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)


def get_more_images(imgs):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []

    for i in range(0, imgs.shape[0]):
        a = imgs[i, :, :, 0]
        b = imgs[i, :, :, 1]
        c = imgs[i, :, :, 2]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)
        bv = cv2.flip(b, 1)
        bh = cv2.flip(b, 0)
        cv = cv2.flip(c, 1)
        ch = cv2.flip(c, 0)

        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)

    more_images = np.concatenate((imgs, v, h))

    return more_images


def duplicate_labels(labels):
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    for i in range(0, labels.shape[0]):
        a = labels[i, :, :, 0]

        av = cv2.flip(a, 1)
        ah = cv2.flip(a, 0)

        vert_flip_imgs.append(av.reshape(IMG_WIDTH, IMG_WIDTH, 1))
        hori_flip_imgs.append(ah.reshape(IMG_WIDTH, IMG_WIDTH, 1))

    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    duplicate_labels = np.concatenate((labels, v, h))
    return duplicate_labels


X_validation = X_train[-50:, ...]
Y_validation = Y_train[-50:]
X_train = X_train[0:-50, ...]
Y_train = Y_train[0:-50]
print("Validation Set of Size " + str(Y_validation.shape[0]) + " Separated")
Y_train.dtype = np.uint8
X_train = get_more_images(X_train)
Y_train = duplicate_labels(Y_train)
print(X_train.shape)
print(Y_train.shape)
print(X_validation.shape)
print(Y_validation.shape)
print("Data Rotation and Flipping Complete")
print(X_train.shape)
print(Y_train.shape)
print(X_validation.shape)
print(Y_validation.shape)
X_train = np.concatenate((X_train, X_validation), axis=0)
Y_train = np.concatenate((Y_train, Y_validation), axis=0)
np.savez_compressed(file='train.npz', X=X_train, Y=Y_train)
np.savez_compressed(file='test.npz', X=X_test)