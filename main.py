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

from tensorflow.keras.callbacks import ModelCheckpoint

from unet.model import unet
from utils.data import train_generator
from utils.data import test_generator
from utils.data import save_results


data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_gen = train_generator(
    2, 'data/membrane/train',
    'image', 'label',
    data_gen_args,
    save_to_dir=None
)

model = unet()
model_checkpoint = ModelCheckpoint(
    'unet_membrane.hdf5',
    monitor='loss',
    verbose=1,
    save_best_only=True
)

model.fit_generator(
    train_gen,
    steps_per_epoch=300,
    epochs=1,
    callbacks=[model_checkpoint]
)

test_gen = test_generator('data/membrane/test')
results = model.predict_generator(test_gen, 30, verbose=True)
save_results('data/membrane/test', results)
