import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torch.nn.utils as utils
import torchvision.transforms as transforms 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
import math


from segmentation_models.condnet.model import CondNet
from segmentation_models.condnet.metrics import mean_squared_error

from uncertainty.metrics import PSNR, SSIM

torch.manual_seed(0)

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

def train_loop(epochs, inputs, targets, slicing = False, dropout_train = False, p = 0.5):
    transformA, transformB = transforms_init(64)
    inputs, targets = transformA(inputs), transformB(targets)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CondNet(p)
    model.to('cpu')
    
    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.01)
    accuracy, losses = [], []
    training = True

    if slicing is not True:
        for epoch in range(0, epochs):
            running_loss = 0
            prediction = model(inputs, dropout_train)
            loss = loss_function(prediction, targets)
            if training is True:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                scheduler.step(loss.item())
            utils.clip_grad_norm_(model.parameters(), max_norm = 2.0, norm_type = 2)
            running_loss += loss.item()
            if math.isnan(loss): 
                raise StopIteration     
            losses.append(loss.item())
        torch.save(losses, 'predictions.pt')
        
    else:
        for epoch in range(0, epochs):
            running_loss = 0
            for idx, slice in enumerate(inputs):
                slice_image, slice_label = inputs[idx:idx + 1, :, :, :], targets[idx:idx + 1, :, :, :]
                prediction = model(slice_image)
                loss = loss_function(prediction, slice_label)

                if training is True:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                scheduler.step(loss.item())
                utils.clip_grad_norm_(model.parameters(), max_norm = 2.0, norm_type = 2)
                running_loss += loss.item()
                if math.isnan(loss): 
                    raise StopIteration
                losses.append(loss) 
    return accuracy, losses


def _train(model, inputs_training, targets_training, loss_function, optimizer, scheduler, num_train, drop_out):
    model.train()
    epoch_train_loss = 0
    psnr, ssim = PSNR(), SSIM()

    slice_image, slice_label = inputs_training[num_train], targets_training[num_train]
    prediction = model(slice_image, drop_out)
    loss = loss_function(prediction, slice_label)

    peaks = psnr(prediction, slice_label)
    ssims = ssim(prediction, slice_label)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    epoch_train_loss += loss.item()

    epoch_train_loss = epoch_train_loss / len(num_train)
    return epoch_train_loss, peaks, ssims

def evaluate(model, inputs_validate, targets_validate, loss_function, num_valid, drop_out):
    model.eval()
    epoch_val_loss = 0

    psnr, ssim = PSNR(), SSIM()

    with torch.no_grad():
        slice_image, slice_label = inputs_validate[num_valid], targets_validate[num_valid]
        prediction = model(slice_image, drop_out)
        loss = loss_function(prediction, slice_label)

        peaks = psnr(prediction, slice_label)
        ssims = ssim(prediction, slice_label)
        epoch_val_loss += loss.item()

    epoch_val_loss =  epoch_val_loss / len(num_valid)
    return epoch_val_loss, peaks, ssims


def train(model, epochs, inputs_training, targets_training, inputs_validate, targets_validate, drop_out = False):
    transformA, transformB = transforms_init(64)
    inputs_training, targets_training = transformA(inputs_training), transformB(targets_training)
    inputs_validate, targets_validate = transformA(inputs_validate), transformB(targets_validate)

    loss_function = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.01)

    train_losses = []
    train_peaks  = []
    train_ssims  = []

    valid_losses = []
    valid_peaks  = []
    valid_ssims  = []

    num_train = np.random.choice(range(len(inputs_training)), size = 10, replace = True)
    num_valid = np.random.choice(range(len(inputs_validate)), size = 10, replace = True)

    for epoch in range(0, epochs):
        train_loss, tr_peaks, tr_ssims = _train(model, inputs_training, targets_training, loss_function, optimizer, scheduler, num_train, drop_out = drop_out)
        valid_loss, vl_peaks, vl_ssims = evaluate(model, inputs_validate, targets_validate, loss_function, num_valid, drop_out = drop_out)

        train_losses.append(train_loss)
        train_peaks .append(tr_peaks)
        train_ssims .append(tr_ssims)

        valid_losses.append(valid_loss)
        valid_peaks .append(vl_peaks)
        valid_ssims .append(vl_ssims)
        
        print(f'Epoch: {epoch + 1 : 02} \t Train Loss: {train_loss:.4f} | PSNR: {tr_peaks:.4f} | SSIM: {tr_ssims:.4f} \t Val. Loss: {valid_loss:.4f} | PSNR: {vl_peaks:.4f} | SSIM: {vl_ssims:.4f}')
    return train_losses, valid_losses, train_peaks, train_ssims, valid_peaks, valid_ssims