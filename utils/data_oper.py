import os  # For filepath, directory handling
import pandas as pd

import cv2  # To read and manipulate images
import numpy as np

import keras.preprocessing.image as idg  # For using image generation
from skimage.morphology import label  # For using image labeling
import matplotlib.pyplot as plt  # Python 2D plotting library
import matplotlib.cm as cm  # Color map


min_object_size = 1


def read_image(filepath, color_mode=cv2.IMREAD_COLOR, target_size=None):
    """Read an image from a file and resize it."""
    img = cv2.imread(filepath, color_mode)
    if target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    return img


def read_test_data_properties(test_dir, img_dir_name):
    """Read basic properties of test images."""
    tmp = []
    for i, dir_name in enumerate(next(os.walk(test_dir))[1]):
        img_dir = os.path.join(test_dir, dir_name, img_dir_name)
        img_name = next(os.walk(img_dir))[2][0]
        img_name_id = os.path.splitext(img_name)[0]
        img_path = os.path.join(img_dir, img_name)
        img_shape = read_image(img_path).shape
        tmp.append(['{}'.format(img_name_id), img_shape[0], img_shape[1],
                    img_shape[0] / img_shape[1], img_shape[2], img_path])

    test_df = pd.DataFrame(tmp, columns=['img_id', 'img_height', 'img_width',
                                         'img_ratio', 'num_channels', 'image_path'])
    return test_df


def normalize_imgs(data):
    """Normalize images."""
    return normalize(data, type_=1)


def normalize_masks(data):
    """Normalize masks."""
    return normalize(data, type_=1)


def normalize(data, type_=1):
    """Normalize data."""
    if type_ == 0:
        # Convert pixel values from [0:255] to [0:1] by global factor
        data = data.astype(np.float32) / data.max()
    if type_ == 1:
        # Convert pixel values from [0:255] to [0:1] by local factor
        div = data.max(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        div[div < 0.01 * data.mean()] = 1.  # protect against too small pixel intensities
        data = data.astype(np.float32) / div
    if type_ == 2:
        # Standardisation of each image
        data = data.astype(np.float32) / data.max()
        mean = data.mean(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        std = data.std(axis=tuple(np.arange(1, len(data.shape))), keepdims=True)
        data = (data - mean) / std

    return data


def trsf_proba_to_binary(y_data):
    """Transform propabilities into binary values 0 or 1."""
    return np.greater(y_data, .5).astype(np.uint8)


def invert_imgs(imgs, cutoff=.5):
    '''Invert image if mean value is greater than cutoff.'''
    imgs = np.array(list(map(lambda x: 1. - x if np.mean(x) > cutoff else x, imgs)))
    return normalize_imgs(imgs)


def imgs_to_grayscale(imgs):
    '''Transform RGB images into grayscale spectrum.'''
    if imgs.shape[3] == 3:
        imgs = normalize_imgs(np.expand_dims(np.mean(imgs, axis=3), axis=3))
    return imgs


def generate_images(imgs, seed=None):
    """Generate new images."""
    # Transformations.
    image_generator = idg.ImageDataGenerator(rotation_range=90.,
                                             width_shift_range=0.02,
                                             height_shift_range=0.02,
                                             zoom_range=0.10,
                                             horizontal_flip=True,
                                             vertical_flip=True)

    # Generate new set of images
    imgs = image_generator.flow(imgs,
                                np.zeros(len(imgs)),
                                batch_size=len(imgs),
                                shuffle=False,
                                seed=seed).next()
    return imgs[0]


def generate_images_and_masks(imgs, masks):
    """Generate new images and masks."""
    seed = np.random.randint(10000)
    imgs = generate_images(imgs, seed=seed)
    masks = trsf_proba_to_binary(generate_images(masks, seed=seed))
    return imgs, masks


# Analyze nuclei sizes.
def get_nuclei_sizes(y_train):
    nuclei_sizes = []
    mask_idx = []
    for i in range(len(y_train)):
        mask = y_train[i].reshape(y_train.shape[1], y_train.shape[2])
        lab_mask = label(mask > .5)
        (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
        nuclei_sizes.extend(mask_sizes[1:])
        mask_idx.extend([i]*len(mask_sizes[1:]))
    return mask_idx, nuclei_sizes

# mask_idx, nuclei_sizes = get_nuclei_sizes()
# nuclei_sizes_df = pd.DataFrame()
# nuclei_sizes_df['mask_index'] = mask_idx
# nuclei_sizes_df['nucleous_size'] = nuclei_sizes
#
# print(nuclei_sizes_df.describe())
# nuclei_sizes_df.sort_values(by='nucleous_size', ascending=True).head(10)



def get_labeled_mask(mask, cutoff=.5):
    """Object segmentation by labeling the mask."""
    mask = mask.reshape(mask.shape[0], mask.shape[1])
    lab_mask = label(mask > cutoff)

    # Keep only objects that are large enough.
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
    if (mask_sizes < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] = 0
        lab_mask = label(lab_mask > cutoff)

    return lab_mask


def get_iou(y_true_labeled, y_pred_labeled):
    """Compute non-zero intersections over unions."""
    # Array of different objects and occupied area.
    (true_labels, true_areas) = np.unique(y_true_labeled, return_counts=True)
    (pred_labels, pred_areas) = np.unique(y_pred_labeled, return_counts=True)

    # Number of different labels.
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    # Each mask has at least one identified object.
    if (n_true_labels > 1) and (n_pred_labels > 1):

        # Compute all intersections between the objects.
        all_intersections = np.zeros((n_true_labels, n_pred_labels))
        for i in range(y_true_labeled.shape[0]):
            for j in range(y_true_labeled.shape[1]):
                m = y_true_labeled[i, j]
                n = y_pred_labeled[i, j]
                all_intersections[m, n] += 1

                # Assign predicted to true background.
        assigned = [[0, 0]]
        tmp = all_intersections.copy()
        tmp[0, :] = -1
        tmp[:, 0] = -1

        # Assign predicted to true objects if they have any overlap.
        for i in range(1, np.min([n_true_labels, n_pred_labels])):
            mn = list(np.unravel_index(np.argmax(tmp), (n_true_labels, n_pred_labels)))
            if all_intersections[mn[0], mn[1]] > 0:
                assigned.append(mn)
            tmp[mn[0], :] = -1
            tmp[:, mn[1]] = -1
        assigned = np.array(assigned)

        # Intersections over unions.
        intersection = np.array([all_intersections[m, n] for m, n in assigned])
        union = np.array([(true_areas[m] + pred_areas[n] - all_intersections[m, n])
                          for m, n in assigned])
        iou = intersection / union

        # Remove background.
        iou = iou[1:]
        assigned = assigned[1:]
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]

        # Labels that are not assigned.
        true_not_assigned = np.setdiff1d(true_labels, assigned[:, 0])
        pred_not_assigned = np.setdiff1d(pred_labels, assigned[:, 1])

    else:
        # in case that no object is identified in one of the masks
        iou = np.array([])
        assigned = np.array([])
        true_labels = true_labels[1:]
        pred_labels = pred_labels[1:]
        true_not_assigned = true_labels
        pred_not_assigned = pred_labels

    # Returning parameters.
    params = {'iou': iou, 'assigned': assigned, 'true_not_assigned': true_not_assigned,
              'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
              'pred_labels': pred_labels}
    return params


def get_score_summary(y_true, y_pred):
    """Compute the score for a single sample including a detailed summary."""

    y_true_labeled = get_labeled_mask(y_true)
    y_pred_labeled = get_labeled_mask(y_pred)

    params = get_iou(y_true_labeled, y_pred_labeled)
    iou = params['iou']
    assigned = params['assigned']
    true_not_assigned = params['true_not_assigned']
    pred_not_assigned = params['pred_not_assigned']
    true_labels = params['true_labels']
    pred_labels = params['pred_labels']
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    summary = []
    for i, threshold in enumerate(np.arange(0.5, 1.0, 0.05)):
        tp = np.sum(iou > threshold)
        fn = n_true_labels - tp
        fp = n_pred_labels - tp
        if (tp + fp + fn) > 0:
            prec = tp / (tp + fp + fn)
        else:
            prec = 0
        summary.append([threshold, prec, tp, fp, fn])

    summary = np.array(summary)
    score = np.mean(summary[:, 1])  # Final score.
    params_dict = {'summary': summary, 'iou': iou, 'assigned': assigned,
                   'true_not_assigned': true_not_assigned,
                   'pred_not_assigned': pred_not_assigned, 'true_labels': true_labels,
                   'pred_labels': pred_labels, 'y_true_labeled': y_true_labeled,
                   'y_pred_labeled': y_pred_labeled}

    return score, params_dict


def get_score(y_true, y_pred):
    """Compute the score for a batch of samples."""
    scores = []
    for i in range(len(y_true)):
        score, _ = get_score_summary(y_true[i], y_pred[i])
        scores.append(score)
    return np.array(scores)


def plot_score_summary(y_true, y_pred):
    """Plot score summary for a single sample."""
    # Compute score and assign parameters.
    score, params_dict = get_score_summary(y_true, y_pred)

    assigned = params_dict['assigned']
    true_not_assigned = params_dict['true_not_assigned']
    pred_not_assigned = params_dict['pred_not_assigned']
    true_labels = params_dict['true_labels']
    pred_labels = params_dict['pred_labels']
    y_true_labeled = params_dict['y_true_labeled']
    y_pred_labeled = params_dict['y_pred_labeled']
    summary = params_dict['summary']

    n_assigned = len(assigned)
    n_true_not_assigned = len(true_not_assigned)
    n_pred_not_assigned = len(pred_not_assigned)
    n_true_labels = len(true_labels)
    n_pred_labels = len(pred_labels)

    # Summary dataframe.
    summary_df = pd.DataFrame(summary, columns=['threshold', 'precision', 'tp', 'fp', 'fn'])
    print('Final score:', score)
    print(summary_df)

    # Plots.
    fig, axs = plt.subplots(2, 3, figsize=(20, 13))

    # True mask with true objects.
    img = y_true
    axs[0, 0].imshow(img, cmap=cm.gray)
    axs[0, 0].set_title('{}.) true mask: {} true objects'.format(n, train_df['num_masks'][n]))

    # True mask with identified objects.
    # img = np.zeros(y_true.shape)
    # img[y_true_labeled > 0.5] = 255
    img, img_type = imshow_args(y_true_labeled)
    axs[0, 1].imshow(img, img_type)
    axs[0, 1].set_title('{}.) true mask: {} objects identified'.format(n, n_true_labels))

    # Predicted mask with identified objects.
    # img = np.zeros(y_true.shape)
    # img[y_pred_labeled > 0.5] = 255
    img, img_type = imshow_args(y_pred_labeled)
    axs[0, 2].imshow(img, img_type)
    axs[0, 2].set_title('{}.) predicted mask: {} objects identified'.format(
        n, n_pred_labels))

    # Prediction overlap with true mask.
    img = np.zeros(y_true.shape)
    img[y_true > 0.5] = 100
    for i, j in assigned: img[(y_true_labeled == i) & (y_pred_labeled == j)] = 255
    axs[1, 0].set_title('{}.) {} pred. overlaps (white) with true objects (gray)'.format(
        n, len(assigned)))
    axs[1, 0].imshow(img, cmap='gray', norm=None)

    # Intersection over union.
    img = np.zeros(y_true.shape)
    img[(y_pred_labeled > 0) & (y_pred_labeled < 100)] = 100
    img[(y_true_labeled > 0) & (y_true_labeled < 100)] = 100
    for i, j in assigned: img[(y_true_labeled == i) & (y_pred_labeled == j)] = 255
    axs[1, 1].set_title('{}.) {} intersections (white) over unions (gray)'.format(
        n, n_assigned))
    axs[1, 1].imshow(img, cmap='gray');

    # False positives and false negatives.
    img = np.zeros(y_true.shape)
    for i in pred_not_assigned: img[(y_pred_labeled == i)] = 255
    for i in true_not_assigned: img[(y_true_labeled == i)] = 100
    axs[1, 2].set_title('{}.) no threshold: {} fp (white), {} fn (gray)'.format(
        n, n_pred_not_assigned, n_true_not_assigned))
    axs[1, 2].imshow(img, cmap='gray');

    # Check the score metric for one sample. The predicted mask is simulated
    # and can be modified in order to check the correct implementation of
    # the score metric.


# Collection of methods for run length encoding.
# For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included
# in the mask. The pixels are one-indexed and numbered from top to bottom,
# then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

def rle_of_binary(x):
    """ Run length encoding of a binary 2D array. """
    dots = np.where(x.T.flatten() == 1)[0]  # indices from top to down
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def mask_to_rle(mask, cutoff=.5, min_object_size=1.):
    """ Return run length encoding of mask. """
    # segment image and label different objects
    lab_mask = label(mask > cutoff)

    # Keep only objects that are large enough.
    (mask_labels, mask_sizes) = np.unique(lab_mask, return_counts=True)
    if (mask_sizes < min_object_size).any():
        mask_labels = mask_labels[mask_sizes < min_object_size]
        for n in mask_labels:
            lab_mask[lab_mask == n] = 0
        lab_mask = label(lab_mask > cutoff)

        # Loop over each object excluding the background labeled by 0.
    for i in range(1, lab_mask.max() + 1):
        yield rle_of_binary(lab_mask == i)


def rle_to_mask(rle, img_shape):
    ''' Return mask from run length encoding.'''
    mask_rec = np.zeros(img_shape).flatten()
    for n in range(len(rle)):
        for i in range(0, len(rle[n]), 2):
            for j in range(rle[n][i + 1]):
                mask_rec[rle[n][i] - 1 + j] = 1
    return mask_rec.reshape(img_shape[1], img_shape[0]).T
