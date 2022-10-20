import os
os.chdir('C:/Users/Renan/Desktop/dsai-thesis')

import torch
import torch.utils.data
from torchvision import transforms
from torchvision.transforms import Normalize

import numpy as np
import matplotlib.pyplot as plt
import logging
from skimage.transform import resize

from segmentation_models.condnet.model import CondNet
from data_loader import load_split_data

def init_condnet(load):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = CondNet()
    if load:
        net.load_state_dict(torch.load(load, map_location = device))
        logging.info(f'Loaded model from {load}')
    
    if device == 'cuda' and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(device)
    return net

def norm_image_tensor(img: torch.Tensor):
    return (img - img.min())/torch.max(img)

def validate_net(net, batch_dict, criterion):
    with torch.no_grad():
        batch_dict = dict(batch_dict)

def slice_all(batch_dict, xslice = 3):
    for tensor in batch_dict:
        batch_dict[tensor] = batch_dict[tensor][xslice].unsqueeze(dim = 0)
    return batch_dict

def image_loader(loader, idx, transform = True):
    if transform == True:
        return loader['T1W'][idx].unsqueeze(dim = 0), loader['T2W'][idx].unsqueeze(dim = 0)
    else:
        return loader['T1W'][idx], loader['T2W'][idx]

def image_size(image, type):
    if type == 1:
        image = image.permute(0, 3, 1, 2)
        image = torch.transpose(image, 0, 1)
    elif type == 2:
        image = torch.transpose(image, 0, 1)
    #assert image.shape[2] == image.shape[3]
    return image

def image_dim(t1w, t2w):
    t1w = t1w[0:t2w.shape[0], :, :, :]
    return t1w, t2w

def view_image(output, slice = 30, transform = False):
    output = output.permute(0, 3, 2, 1)
    output = output.detach().cpu().numpy()
    if transform is True:
        pass
    center = output.shape[0]//2
    plt.imshow(output[slice, :, :]) 
    plt.colorbar()

def tensorboard_write(loss, val_loss, model, writer):
    writer.add_scalars(f'loss', {
        'Train': loss.item(),
        'Validation': val_loss.item()})

def view_prediction(prediction):
    import numpy as np
    transform = transforms.Compose([
        transforms.Normalize(
            mean=[0.485],
            std =[0.229])])
    image = transform(prediction).permute(2, 3, 0, 1).detach().numpy()
    plt.imshow(np.abs(image[:, :, 9, 0]), cmap = 'inferno')
    plt.colorbar()

class ColeColeModel():
    def __init__(self):
        super(ColeColeModel, self).__init__()
        self.blood = 0.700
        self.bone_canc = 0.080
        self.bone_cort = 0.020
        self.cerebellum = 0.130
        self.csf = 2.00
        self.dura = 0.500
        self.fat = 0.040
        self.gm = 0.100
        self.mtissue = 0.070
        self.muscle = 0.340
        self.skin = 0.100
        self.vhumor = 1.500
        self.wm = 0.070
        self.tissues = [self.blood, self.bone_canc, self.bone_cort, self.cerebellum,
                                self.csf, self.dura, self.fat, self.gm, self.mtissue,
                                self.muscle, self.skin, self.vhumor, self.wm]

    def __getitem__(self, index):
        true = []
        labels = ['blood', 'bone_canc', 'bone_cort', 'cerebellum',
                    'csf', 'dura', 'fat', 'gm', 'mtissue',
                    'muscle', 'skin', 'vhumor', 'wm']
        for tissue in self.tissues:
            true.append(tissue)
        return true.__getitem__(self, index)

class ToTensor:
    def __call__(self, sample):
        for key in sample:
            sample[key] = torch.from_numpy(sample[key].astype(np.float32))
        return sample

class Norm:
    def __init__(self, mean, std):
        self.normalize = Normalize(mean, std)

    def __call__(self, image):
        return self.normalize(image)


def mpl_image_grid(images):
    n = min(images.shape[0], 16) 
    rows = 4
    cols = (n // 4) + (1 if (n % 4) != 0 else 0)
    figure = plt.figure(figsize=(2 * rows, 2 * cols))
    plt.subplots_adjust(0, 0, 1, 1, 0.001, 0.001)
    for i in range(n):
        plt.subplot(cols, rows, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        if images.shape[1] == 3:
            vol = images[i].detach().numpy()
            img = [[[(1-vol[0,x,y]) * vol[1, x, y], (1 - vol[0, x, y]) * vol[2, x, y], 0] \
                            for y in range(vol.shape[2])] \
                            for x in range(vol.shape[1])]
            plt.imshow(img)
        else: 
            plt.imshow((images[i, 0] * 255).int(), cmap = 'gray')
    return figure

def prediction_tensorboard(prediction):
    image = prediction.permute(2, 3, 0, 1).detach().cpu().numpy()
    figure = plt.imshow(np.abs(image[:, :, 0, 0]), cmap = 'inferno')
    return figure

def transforms_init(size):
    transformA = transforms.Compose([ 
                        transforms.Normalize(
                                mean = [0.485],
                                std  = [0.229]),
                        transforms.Resize(size)    
                        ])
    transformB = transforms.Compose([ 
                        transforms.Normalize(
                                mean = [0.485],
                                std  = [0.229]),
                        transforms.Resize(size - 7)
                        ])
    return transformA, transformB

def log_to_tensorboard(writer, loss, slice_image, target, prediction, counter):
    writer.add_scalar('Loss', loss, counter)
    writer.add_figure('MRI T1W data', prediction_tensorboard(slice_image), global_step = counter)
    writer.add_figure('Predicted data', prediction_tensorboard(prediction), global_step = counter)
    writer.add_figure('Image data', mpl_image_grid(slice_image.float().cpu()), global_step = counter)
    writer.add_figure("Mask", mpl_image_grid(target.float().cpu()), global_step = counter)
    writer.add_figure('Prediction', 
                        mpl_image_grid(torch.argmax(prediction.cpu(), dim = 1, keepdim=True)), 
                        global_step=counter)