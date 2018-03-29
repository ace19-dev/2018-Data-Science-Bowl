import cv2
import numpy as np

import tensorflow as tf


IMG_WIDTH = 256
IMG_HEIGHT = 256


def image_augmentation(image, mask):
    """Returns (maybe) augmented images
    (1) Random flip (left <--> right)
    (2) Random flip (up <--> down)
    (3) Random brightness
    (4) Random hue
    Args:
        image (3-D Tensor): Image tensor of (H, W, C)
        mask (3-D Tensor): Mask image tensor of (H, W, 1)
    Returns:
        image: Maybe augmented image (same shape as input `image`)
        mask: Maybe augmented mask (same shape as input `mask`)
    """
    concat_image = tf.concat([image, mask], axis=-1)

    maybe_flipped = tf.image.random_flip_left_right(concat_image)
    maybe_flipped = tf.image.random_flip_up_down(concat_image)

    image = maybe_flipped[:, :, :-1]
    mask = maybe_flipped[:, :, -1:]

    image = tf.image.random_brightness(image, 0.7)
    image = tf.image.random_hue(image, 0.3)

    return image, mask


# def get_image_mask(queue, augmentation=True):
#     """Returns `image` and `mask`
#     Input pipeline:
#         Queue -> CSV -> FileRead -> Decode JPEG
#     (1) Queue contains a CSV filename
#     (2) Text Reader opens the CSV
#         CSV file contains two columns
#         ["path/to/image.jpg", "path/to/mask.jpg"]
#     (3) File Reader opens both files
#     (4) Decode JPEG to tensors
#     Notes:
#         height, width = 640, 960
#     Returns
#         image (3-D Tensor): (640, 960, 3)
#         mask (3-D Tensor): (640, 960, 1)
#     """
#     text_reader = tf.TextLineReader(skip_header_lines=1)
#     _, csv_content = text_reader.read(queue)
#
#     image_path, mask_path = tf.decode_csv(
#         csv_content, record_defaults=[[""], [""]])
#
#     image_file = tf.read_file(image_path)
#     mask_file = tf.read_file(mask_path)
#
#     image = tf.image.decode_jpeg(image_file, channels=3)
#     image.set_shape([height, width, 3])
#     image = tf.cast(image, tf.float32)
#
#     mask = tf.image.decode_jpeg(mask_file, channels=1)
#     mask.set_shape([height, width, 1])
#     mask = tf.cast(mask, tf.float32)
#     mask = mask / (tf.reduce_max(mask) + 1e-7)
#
#     if augmentation:
#         image, mask = image_augmentation(image, mask)
#
#     return image, mask


def get_more_images(imgs):
    # more_images = []
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


def augment_brightness_camera_images(image):
  ### Augment brightness
  image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
  random_bright = .25 + np.random.uniform()
  # print(random_bright)
  image1[:, :, 2] = image1[:, :, 2] * random_bright
  image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
  return image1


def trans_image(image, bb_boxes_f, trans_range):
  # Translation augmentation
  bb_boxes_f = bb_boxes_f.copy(deep=True)

  tr_x = trans_range * np.random.uniform() - trans_range / 2
  tr_y = trans_range * np.random.uniform() - trans_range / 2

  Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
  rows, cols, channels = image.shape
  bb_boxes_f['xmin'] = bb_boxes_f['xmin'] + tr_x
  bb_boxes_f['xmax'] = bb_boxes_f['xmax'] + tr_x
  bb_boxes_f['ymin'] = bb_boxes_f['ymin'] + tr_y
  bb_boxes_f['ymax'] = bb_boxes_f['ymax'] + tr_y

  image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))

  return image_tr, bb_boxes_f


def stretch_image(img, bb_boxes_f, scale_range):
  # Stretching augmentation

  bb_boxes_f = bb_boxes_f.copy(deep=True)

  tr_x1 = scale_range * np.random.uniform()
  tr_y1 = scale_range * np.random.uniform()
  p1 = (tr_x1, tr_y1)
  tr_x2 = scale_range * np.random.uniform()
  tr_y2 = scale_range * np.random.uniform()
  p2 = (img.shape[1] - tr_x2, tr_y1)

  p3 = (img.shape[1] - tr_x2, img.shape[0] - tr_y2)
  p4 = (tr_x1, img.shape[0] - tr_y2)

  pts1 = np.float32([[p1[0], p1[1]],
                     [p2[0], p2[1]],
                     [p3[0], p3[1]],
                     [p4[0], p4[1]]])
  pts2 = np.float32([[0, 0],
                     [img.shape[1], 0],
                     [img.shape[1], img.shape[0]],
                     [0, img.shape[0]]]
                    )

  M = cv2.getPerspectiveTransform(pts1, pts2)
  img = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]))
  img = np.array(img, dtype=np.uint8)

  bb_boxes_f['xmin'] = (bb_boxes_f['xmin'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
  bb_boxes_f['xmax'] = (bb_boxes_f['xmax'] - p1[0]) / (p2[0] - p1[0]) * img.shape[1]
  bb_boxes_f['ymin'] = (bb_boxes_f['ymin'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]
  bb_boxes_f['ymax'] = (bb_boxes_f['ymax'] - p1[1]) / (p3[1] - p1[1]) * img.shape[0]

  return img, bb_boxes_f


# How to use augment func
# e.g.
# def get_image_name(df, ind, size=(640, 300), augmentation=False, trans_range=20, scale_range=20):
#   ### Get image by name
#
#   file_name = df['File_Path'][ind]
#   img = cv2.imread(file_name)
#   img_size = np.shape(img)
#
#   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#   img = cv2.resize(img, size)
#   name_str = file_name.split('/')
#   name_str = name_str[-1]
#   # print(name_str)
#   # print(file_name)
#   bb_boxes = df[df['Frame'] == name_str].reset_index()
#   img_size_post = np.shape(img)
#
#   if augmentation == True:
#     img, bb_boxes = trans_image(img, bb_boxes, trans_range)
#     img, bb_boxes = stretch_image(img, bb_boxes, scale_range)
#     img = augment_brightness_camera_images(img)
#
#   bb_boxes['xmin'] = np.round(bb_boxes['xmin'] / img_size[1] * img_size_post[1])
#   bb_boxes['xmax'] = np.round(bb_boxes['xmax'] / img_size[1] * img_size_post[1])
#   bb_boxes['ymin'] = np.round(bb_boxes['ymin'] / img_size[0] * img_size_post[0])
#   bb_boxes['ymax'] = np.round(bb_boxes['ymax'] / img_size[0] * img_size_post[0])
#   bb_boxes['Area'] = (bb_boxes['xmax'] - bb_boxes['xmin']) * (bb_boxes['ymax'] - bb_boxes['ymin'])
#   # bb_boxes = bb_boxes[bb_boxes['Area']>400]
#
#   return name_str, img, bb_boxes