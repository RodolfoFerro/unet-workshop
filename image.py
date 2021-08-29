# ==============================================================
# Author: Rodolfo Ferro
# Twitter: @FerroRodolfo
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script has been originally created by Rodolfo Ferro.
# Any explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ==============================================================

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import cv2

import os


def load_test_image(img_index, folder='data/membrane/test/', plot_img=True):
    img_path = os.path.join(folder, f'{img_index}.png')
    img = cv2.imread(folder + '0.png', 0)
    img = cv2.resize(img, (256, 256))
    img = img.astype(np.uint)

    if plot_img:
        plt.imshow(img, cmap='gray')
        plt.axis('off')

    return img


def inference_over_image(model, img, plot_img=True):
    input_img = img.reshape((1, img.shape[0], img.shape[1], 1))
    out = model.predict(input_img)
    out = out.reshape((img.shape[0], img.shape[1]))
    out *= 255
    out = out.astype(np.uint)

    if plot_img:
        plt.imshow(out, cmap='gray')
        plt.axis('off')
    
    return out


def create_mask(out_img, plot_img=True):
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