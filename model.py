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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import Adam


def unet(pretrained_weights=None, input_size=(256, 256, 1)):
    """U-Net model constructor.

    Parameters
    ----------
    pretrained_weights : str
        Path to pretrained weights.
    input_size : tuple
        Spatial size of the expected input image.
    """

    inputs = Input(input_size)

    # Convolution chain #1
    conv_1 = Conv2D(
        64, 3,
        activation='relu', 
        padding='same',
        kernel_initializer='he_normal'
    )(inputs)
    conv_1 = Conv2D(
        64, 3,
        activation='relu',
        padding='same', 
        kernel_initializer='he_normal'
    )(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_1)

    # Convolution chain #2
    conv_2 = Conv2D(
        128, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(pool_1)
    conv_2 = Conv2D(
        128, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_2)
    pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_2)

    # Convolution chain #3
    conv_3 = Conv2D(
        256, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(pool_2)
    conv_3 = Conv2D(
        256, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_3)
    pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_3)

    # Convolution chain #4
    conv_4 = Conv2D(
        512, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(pool_3)
    conv_4 = Conv2D(
        512, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_4)
    drop_4 = Dropout(0.5)(conv_4)
    pool_4 = MaxPooling2D(pool_size=(2, 2))(drop_4)

    # Convolution chain #5 - Middle part + Dropout
    conv_5 = Conv2D(
        1024, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(pool_4)
    conv_5 = Conv2D(
        1024, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_5)
    drop_5 = Dropout(0.5)(conv_5)

    # Up-sampling chain #1
    upsample_1 = UpSampling2D(size=(2, 2))(drop_5)
    up_6 = Conv2D(
        512, 2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(upsample_1)
    concat_6 = concatenate([drop_4, up_6], axis=3)
    conv_6 = Conv2D(
        512, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(concat_6)
    conv_6 = Conv2D(
        512, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_6)

    # Up-sampling chain #2
    upsample_2 = UpSampling2D(size=(2, 2))(conv_6)
    up_7 = Conv2D(
        256, 2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(upsample_2)
    concat_7 = concatenate([conv_3, up_7], axis=3)
    conv_7 = Conv2D(
        256, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(concat_7)
    conv_7 = Conv2D(
        256, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_7)

    # Up-sampling chain #3
    upsample_3 = UpSampling2D(size=(2, 2))(conv_7)
    up_8 = Conv2D(
        128, 2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(upsample_3)
    concat_8 = concatenate([conv_2, up_8], axis=3)
    conv_8 = Conv2D(
        128, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(concat_8)
    conv_8 = Conv2D(
        128, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_8)

    # Up-sampling chain #4
    upsample_4 = UpSampling2D(size=(2, 2))(conv_8)
    up_9 = Conv2D(
        64, 2,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(upsample_4)
    concat_9 = concatenate([conv_1, up_9], axis=3)
    conv_9 = Conv2D(
        64, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(concat_9)
    conv_9 = Conv2D(
        64, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_9)
    conv_9 = Conv2D(
        2, 3,
        activation='relu',
        padding='same',
        kernel_initializer='he_normal'
    )(conv_9)
    conv_10 = Conv2D(
        1, 1,
        activation='sigmoid'
    )(conv_9)

    model = Model(inputs=inputs, outputs=conv_10)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
