# ==============================================================
# Author: Rodolfo Ferro
# Twitter: @rodo_ferro
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script has been originally created by Rodolfo Ferro.
# Any explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ==============================================================

# -*- coding: utf-8 -*-

import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import skimage.transform as transf
import skimage.io as io
import numpy as np

sky = [128, 128, 128]
building = [128, 0, 0]
pole = [192, 192, 128]
road = [128, 64, 128]
pavement = [60, 40, 222]
tree = [128, 128, 0]
sign_symbol = [192, 128, 128]
fence = [64, 64, 128]
car = [64, 0, 128]
pedestrian = [64, 64, 0]
bicyclist = [0, 128, 192]
unlabelled = [0, 0, 0]

COLOR_DICT = np.array([
    sky, building, pole, road, pavement, tree, sign_symbol, fence, car,
    pedestrian, bicyclist, unlabelled
])


def adjust_data(img, mask, flag_multi_class, num_class):
    """Adjust data function.
    
    Parameters
    ----------
    img : array
        Image array.
    mask : array
        Mask array.
    flag_multi_class : bool
        Flag for multi-class segmentation.
    num_class : int
        Number of classes.
    
    Returns
    -------
    img : array
        Image array.
    mask : array
        Mask array.
    """

    if flag_multi_class:
        img = img / 255.
        mask = mask[:, :, :, 0] if len(mask.shape) == 4 else mask[:, :, 0]
        new_mask = np.zeros(mask.shape + (num_class, ))

        for i in range(num_class):
            new_mask[mask == i, i] = 1

        new_mask = np.reshape(
            new_mask,
            (new_mask.shape[0], new_mask.shape[1] * new_mask.shape[2],
             new_mask.shape[3])) if flag_multi_class else np.reshape(
                 new_mask,
                 (new_mask.shape[0] * new_mask.shape[1], new_mask.shape[2]))
        mask = new_mask

    elif np.max(img) > 1:
        img = img / 255.
        mask = mask / 255.
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0

    return img, mask


def train_generator(batch_size,
                    train_path,
                    image_folder,
                    mask_folder,
                    aug_dict,
                    image_color_mode='grayscale',
                    mask_color_mode='grayscale',
                    image_save_prefix='image',
                    mask_save_prefix='mask',
                    flag_multi_class=False,
                    num_class=2,
                    save_to_dir=None,
                    target_size=(256, 256),
                    seed=1):
    """Train generator function.
    
    Parameters
    ----------
    batch_size : int
        Batch size.
    train_path : str
        Path to training data.
    image_folder : str
        Path of the folder containing the images.
    mask_folder : str
        Path of the folder containing the masks.
    aug_dict : dict
        Dictionary of augmentation parameters.
    image_color_mode : str, optional
        Color mode for the images. The default is 'grayscale'.
    mask_color_mode : str, optional
        Color mode for the masks. The default is 'grayscale'.
    image_save_prefix : str, optional
        Prefix for the images. The default is 'image'.
    mask_save_prefix : str, optional
        Prefix for the masks. The default is 'mask'.
    flag_multi_class : bool, optional
        Flag for multi-class segmentation. The default is False.
    num_class : int, optional
        Number of classes. The default is 2.
    save_to_dir : str, optional
        Path to save the augmented images. The default is None.
    target_size : tuple, optional
        Target size for the images. The default is (256, 256).
    seed : int, optional
        Seed for the random number generator. The default is 1.
    
    Yields
    ------
    img : array
        Image array.
    mask : array
        Mask array.
    """

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=image_save_prefix,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)

    train_gen = zip(image_generator, mask_generator)
    for img, mask in train_gen:
        img, mask = adjust_data(img, mask, flag_multi_class, num_class)
        yield img, mask


def test_generator(test_path,
                   num_image=30,
                   target_size=(256, 256),
                   flag_multi_class=False,
                   as_gray=True):
    """Test generator function.

    Parameters
    ----------
    test_path : str
        Path to test data.
    num_image : int, optional
        Number of images. The default is 30.
    target_size : tuple, optional
        Target size for the images. The default is (256, 256).
    flag_multi_class : bool, optional
        Flag for multi-class segmentation. The default is False.
    as_gray : bool, optional
        Flag for grayscale images. The default is True.
    
    Yields
    ------
    img : array
        Image array.
    """

    for i in range(num_image):
        img_path = os.path.join(test_path, f'{i}.png')
        img = io.imread(img_path, as_gray=as_gray)
        img = img / 255.
        img = transf.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) \
            if not flag_multi_class else img
        img = np.reshape(img, (1, ) + img.shape)
        yield img


def visualize_label(num_class, color_dict, img):
    """Visualize label function.

    Parameters
    ----------
    num_class : int
        Number of classes.
    color_dict : dict
        Dictionary of colors.
    img : array
        Image array.
    
    Returns
    -------
    img_out : array
        Image array.
    """

    img = img[:, :, 0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3, )).astype(np.uint8)

    for i in range(num_class):
        img_out[img == i, :] = color_dict[i]

    return img_out / 255.


def save_results(save_path, npyfile, flag_multi_class=False, num_class=2):
    """Save results function.

    Parameters
    ----------
    save_path : str
        Path to save the results.
    npyfile : array
        Array of results.
    flag_multi_class : bool, optional
        Flag for multi-class segmentation. The default is False.
    num_class : int, optional
        Number of classes. The default is 2.
    """

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i, img in enumerate(npyfile):
        # img = visualize_label(num_class, COLOR_DICT, item) \
        #     if flag_multi_class else item[:, :, 0]

        img *= 255.
        img = img.astype(np.uint8)

        img_path = os.path.join(save_path, f'{i}.png')
        io.imsave(img_path, img)
