'''
Module loads the dataset into memory
'''

import numpy as np
import os
from os import listdir
from os.path import isfile, join
from medpy.io import load


def med_reshape(image, new_shape):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[:image.shape[0], :image.shape[1], :image.shape[1]] = reshaped_image
    return reshaped_image

def load_data(y_shape, z_shape):
    image_dir = os.path.join('dataset/images', 'train')
    label_dir = os.path.join('dataset', 'labels')

    images = [f for f in listdir(image_dir) if (isfile(join(image_dir, f)) and f[0] != '.')]
    data = []
    y_shape = 64
    z_shape = 64
    for f in images[0:20]:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))
        image = image.astype('float')
        image /= np.max(image)

        image = med_reshape(image, new_shape = (image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape = (label.shape[0], y_shape, z_shape)).astype(int)
        data.append({'image': image, 'seg': label, 'filename': f})
    return np.array(data)