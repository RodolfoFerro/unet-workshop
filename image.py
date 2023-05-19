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

import matplotlib.pyplot as plt
import numpy as np
import cv2


def load_test_image(img_index=0, folder='data/membrane/test/', plot_img=True):
    """Load test image function.
    
    Parameters
    ----------
    img_index : int (optional)
        Index of image to load. Default is 0.
    folder : str (optional)
        Path to folder containing images.
    plot_img : bool (optional)
        Flag to plot image. Default is True.
    
    Returns
    -------
    img : array
        Image array.
    """

    img_path = os.path.join(folder, f'{img_index}.png')
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.uint)

    if plot_img:
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    return img


def inference_over_image(model, img, plot_img=True):
    """Inference over image function.

    Parameters
    ----------
    model : Keras model
        Keras model.
    img : array
        Image array.
    plot_img : bool (optional)
        Flag to plot image. Default is True.
    
    Returns
    -------
    out : array
        Output array.
    """

    input_img = img.reshape((1, img.shape[0], img.shape[1], 1))
    out = model.predict(input_img)
    out = out.reshape((img.shape[0], img.shape[1]))
    out *= 255
    out = out.astype(np.uint)

    if plot_img:
        plt.imshow(out, cmap='gray')
        plt.axis('off')

    return out


def create_mask(out, plot_img=True):
    """Create mask function.

    Parameters
    ----------
    out : array
        Output array.
    plot_img : bool (optional)
        Flag to plot image. Default is True.

    Returns
    -------
    mask : array
        Mask array.
    """

    mask = np.zeros((out.shape[0], out.shape[1], 3), np.uint)
    mask[:, :, 0] = out
    mask[:, :, 1] = out
    mask[:, :, 2] = out

    mask = cv2.bitwise_not(mask) + 256
    mask[:, :, 1] = 0
    mask[:, :, 2] = 0
    mask.astype(np.uint)

    if plot_img:
        plt.imshow(mask, cmap='gray')
        plt.axis('off')

    return mask


def overlay_mask(img, mask, plot_img=True):
    """Overlay mask function.

    Parameters
    ----------
    img : array
        Image array.
    mask : array
        Mask array.
    plot_img : bool (optional)
        Flag to plot image. Default is True.

    Returns
    -------
    res : array
        Resulting array.
    """

    color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint)
    color_img[:, :, 0] = img
    color_img[:, :, 1] = img
    color_img[:, :, 2] = img
    color_img = color_img.astype(np.uint)
    res = cv2.bitwise_or(color_img, mask)

    if plot_img:
        plt.imshow(res)
        plt.axis('off')

    return res
