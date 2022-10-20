'''
Module loads the dataset into memory
'''
import numpy as np
import os
from os import listdir
from os.path import isfile, join
from medpy.io import load
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader
from slicesdataset import SlicesDataset

def med_reshape(image, new_shape):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[:image.shape[0], :image.shape[1], :image.shape[1]] = reshaped_image
    return reshaped_image

def load_data(y_shape = 64, z_shape = 64):
    image_dir = os.path.join('dataset/images', 'train')
    label_dir = os.path.join('dataset', 'labels')

    images = [f for f in listdir(image_dir) if (isfile(join(image_dir, f)) and f[0] != '.')]
    data = {}

    for f in images[0:20]:
        image, _ = load(os.path.join(image_dir, f))
        label, _ = load(os.path.join(label_dir, f))
        image = image.astype('float')
        image /= np.max(image)

        image = med_reshape(image, new_shape = (image.shape[0], y_shape, z_shape))
        label = med_reshape(label, new_shape = (label.shape[0], y_shape, z_shape)).astype(int)
        data['image'] = image
        data['seg'] = label
        data['filename'] = f
    return data 


def split_init():
    os.chdir('C:/Users/Renan/Desktop/dsai-thesis')
    directory = 'data_simon/SIMON_BIDS/sub-032633'
    FOLDERS = []
    FOLDERS_PATH = []
    COMPLETE_DATA = []
    COMPLETE_PATH = []

    for folder in os.listdir(directory):
        FOLDERS.append(folder)
        FOLDERS_PATH.append(os.path.join(directory, folder))

    for folder in FOLDERS_PATH:
        if len(os.listdir(folder)) > 1:
            path = os.path.join(folder, 'anat')
            ITEMS = []
            for item in os.listdir(path):
                ITEMS.append(item)
                for item in ITEMS:
                    if item.endswith('T2star.nii.gz'):
                        COMPLETE_DATA.append(folder)
    COMPLETE_DATA = list(set(COMPLETE_DATA))                 
    for complete in COMPLETE_DATA:
        COMPLETE_PATH.append(os.path.join(complete, 'anat'))

    T1W_FILES = []
    T2W_FILES = []
    T2STAR_FILES = []
    for folder in COMPLETE_PATH:
        for items in os.listdir(folder):
            if items.endswith('run-1_T1w.nii.gz') and not items.endswith('acq-10iso_run-1_T1w.nii.gz'):
                T1W_FILES.append(items)
            elif items.endswith('run-1_T2w.nii.gz'):
                T2W_FILES.append(items)
            elif items.endswith('run-1_T2star.nii.gz') and not items.endswith('acq-ph_run-1_T2star.nii.gz') and not items.endswith('acq-pmri_run-1_T2star.nii.gz'):
                T2STAR_FILES.append(items)

    T1W_PATH, T2W_PATH, T2STAR_PATH = [], [], []
    for T1W, T2W, T2STAR, complete in zip(T1W_FILES, T2W_FILES, T2STAR_FILES, COMPLETE_DATA):
        T1W_PATH.append(os.path.join(complete, 'anat', T1W))
        T2W_PATH.append(os.path.join(complete, 'anat', T2W))
        T2STAR_PATH.append(os.path.join(complete, 'anat', T2STAR))
    assert len(T1W_PATH) and len(T2W_PATH) and len(T2STAR_PATH)

    T1W_IMAGES, T2W_IMAGES, T2S_IMAGES = {}, {}, {}

    for complete, t1w, t2w, t2s in zip(COMPLETE_PATH, T1W_PATH, T2W_PATH, T2STAR_PATH):
        path1 = 'data_simon/SIMON_BIDS/sub-032633/ses'
        path2 = '/anat/sub-032633_ses'

        T1W_IMAGES['T1W' + complete[36:40]] = path1 + complete[36:40] + path2 + complete[36:40] + '_run-1_T1w.nii.gz'
        T2W_IMAGES['T2W' + complete[36:40]] = path1 + complete[36:40] + path2 + complete[36:40] + '_run-1_T2w.nii.gz'
        T2S_IMAGES['T2S' + complete[36:40]] = path1 + complete[36:40] + path2 + complete[36:40] + '_run-1_T2star.nii.gz'

    T1W_KEYS, T2W_KEYS, T2S_KEYS = [], [], []

    for T1W, T2W, T2S in zip(T1W_IMAGES.keys(), T2W_IMAGES.keys(), T2S_IMAGES.keys()):
        T1W_KEYS.append(T1W[4:7])
        T2W_KEYS.append(T2W[4:7])
        T2S_KEYS.append(T2S[4:7])
    T1W_KEYS = T1W_KEYS[1:4]
    T2W_KEYS = T2W_KEYS[1:4]
    T2S_KEYS = T2S_KEYS[1:5] 

    # assert T1W_KEYS == T2W_KEYS == T2S_KEYS

    split = dict()
    split['train'] = T1W_KEYS[0 : int(0.7 * len(T1W_KEYS))] 
    split['val'] = T1W_KEYS[int(0.7 * len(T1W_KEYS)) : int(0.9 * len(T1W_KEYS))]
    split['test'] = T1W_KEYS[int(0.9 * len(T1W_KEYS)) : ]

    assert(not bool(set(split['train']) & set(split['val'])))
    assert(not bool(set(split['val']) & set(split['test'])))
    return split, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES

def load_image(filename: str, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES):
    t1_weighted = sitk.ReadImage(T1W_IMAGES['T1W-' + filename], sitk.sitkFloat32)
    t2_weighted = sitk.ReadImage(T2W_IMAGES['T2W-' + filename], sitk.sitkFloat32)
    t2_star = sitk.ReadImage(T2S_IMAGES['T2S-' + filename], sitk.sitkFloat32)
    return t1_weighted, t2_weighted, t2_star

def load_split_data(data_type, preprocess = True):
    split, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES = split_init()
    T1W_load, T2W_load, T2S_load = [], [], []
    for data in split[data_type]:
        T1W, T2W, T2S = load_image(data, T1W_IMAGES, T2W_IMAGES, T2S_IMAGES)
        if preprocess == True:
            T1W, T2W, T2S = torch.tensor(sitk.GetArrayFromImage(T1W)), torch.tensor(sitk.GetArrayFromImage(T2W)), torch.tensor(sitk.GetArrayFromImage(T2S))
        T1W_load.append(T1W), T2W_load.append(T2W), T2S_load.append(T2S) 
    type_loader = {}
    type_loader['T1W'] = T1W_load
    type_loader['T2W'] = T2W_load
    type_loader['T2S'] = T2S_load
    return type_loader

def create_mat(cond):
    mat1 = np.array([0]*(1 * 249 * 249), dtype = float).reshape(1, 249, 249)
    for i in range(mat1.shape[0]):
        for j in range(mat1.shape[1]):
            mat1[i][j] = cond
    return mat1

def gen_target():
    conductivity = [0.700, 0.080, 0.020, 0.130, 2.000, 
                        0.500, 0.040, 0.100, 0.070, 0.340]
    cond1, cond2, cond3, cond4, cond5 = create_mat(0.700), create_mat(0.080), create_mat(0.020), create_mat(0.130), create_mat(2.000)
    cond6, cond7, cond8, cond9, cond0 = create_mat(0.500), create_mat(0.040), create_mat(0.100), create_mat(0.070), create_mat(0.340)
    targets = np.concatenate((cond1, cond2, cond3, cond4, cond5, cond6, cond7, cond8, cond9, cond0), axis = 0)
    targets = torch.tensor(targets)
    targets = targets.unsqueeze(1)
    return targets.type(torch.FloatTensor)

def load_dir(type: 'str' = 'train'):
    os.chdir('C:/Users/Renan/Desktop/dsai-thesis')
    image_dir, label_dir = os.path.join(type, 'inputs'), os.path.join(type, 'labels')
    IMAGES, LABELS = [], []
    IMAGES_DIR, LABELS_DIR = [], []
    for item_img, item_lbl in zip(os.listdir(image_dir), os.listdir(label_dir)):
        IMAGES.append(item_img)
        LABELS.append(item_lbl)
    
    for img, lbl in zip(IMAGES, LABELS):
        IMAGES_DIR.append(os.path.join(image_dir, img))
        LABELS_DIR.append(os.path.join(label_dir, lbl))
    return IMAGES_DIR, LABELS_DIR

def shuffle_split_data(X, y):
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, 70)
    X_train = X[split]
    y_train = y[split]
    X_val   = X[~split]
    y_val   = y[~split]
    return X_train, y_train, X_val, y_val

def load_dataset(type: 'str' = 'train', batch_size = 500):
    IMAGES_DIR, LABELS_DIR = load_dir(type)
    # X = torch.empty((176, 1, 256, 256), dtype = torch.float32)
    # y = torch.empty((176, 1, 249, 249), dtype = torch.float32)
    X, y = [], []
    for data, label in zip(IMAGES_DIR, LABELS_DIR):
        image = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(data, sitk.sitkFloat32)))
        image = image.unsqueeze(0)
        image = image.permute(3, 0, 1, 2)
        # X = torch.concat((X, image[:176, :, :, :]), dim = 0)
        X.append(image[:176, :, :, :])

        labels = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(label, sitk.sitkFloat32)))
        labels = labels.unsqueeze(0)
        labels = labels.permute(3, 0, 1, 2)
        # y = torch.concat((y, labels[:176, :, :249, :249]), dim = 0)
        y.append(labels[:176, :, :249, :249])
    
    train_data = []
    for i in range(len(X)):
        train_data.append([X[i], y[i]])
    dataloader = DataLoader(train_data, shuffle = True, batch_size = batch_size)
    X, y = next(iter(dataloader))
    X = X.reshape((X.shape[0] * X.shape[1], 1, 256, 256))
    y = y.reshape((y.shape[0] * y.shape[1], 1, 249, 249))
    return X, y

def threshold(image):
    '''
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box and compute the background 
    median intensity.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
        Background median intensity value.
    '''
    inside_value  = 0
    outside_value = 255
    bin_image = sitk.OtsuThreshold(image, inside_value, outside_value)

    label_intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats_filter.SetBackgroundValue(outside_value)
    label_intensity_stats_filter.Execute(bin_image,image)
    bg_mean = label_intensity_stats_filter.GetMedian(inside_value)
    
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()    
    label_shape_filter.Execute(bin_image)
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    return sitk.RegionOfInterest(image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])