import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F
import torch.nn.utils as utils
import torchvision.transforms as transforms 
import os
os.chdir('C:/Users/Renan/Desktop/dsai-thesis')

from segmentation_models.condnet.model import CondNet
from segmentation_models.condnet.train import train, transforms_init

from uncertainty.sdenet import SDENetClassification, SDENetRegression
from uncertainty.metrics import uncertainty_metrics
from uncertainty.utils import get_output


def optimizer_init(model):
    optimizer_F = optim.SGD([ 
                         {'params': model.downsampling_layers.parameters()}, 
                         {'params': model.drift.parameters()},
                         {'params': model.fc_layers.parameters()}], 

                         lr = 0.001, weight_decay = 5e-4)

    optimizer_G = optim.SGD([ 
                         {'params': model.diffusion.parameters()}], 
                         
                         lr = 0.001, weight_decay = 5e-4)
    return optimizer_F, optimizer_G

def sdenet_train(epochs, inputs, targets, depth = 6, display = False):
    conductivity_model, uncertainty_model = CondNet(), SDENetRegression(depth)
    conductivity_model.to('cpu'), uncertainty_model.to('cpu')
    conductivity_optimizer = optim.Adam(conductivity_model.parameters(), lr = 0.001)
    optimizer_F, optimizer_G = optimizer_init(uncertainty_model)
    loss_function = torch.nn.MSELoss()
    losses_condnet, losses_sdenet = [], []

    m, n = 128, 10
    transformA, transformB = transforms_init(m)
    num = 2200
    print(f'Training CondNet & SDENet at index {num} - {num + n}')

    X, y = inputs[num:num+n, :, :, :], targets[num:num+n, :, :, :]
    X_cond, X_unc, y_unc, y_cond = transformA(X), transformA(X), transformA(y), transformB(y)
    X_unc, y_unc = X_unc.permute(0, 2, 3, 1), y_unc.permute(0, 2, 3, 1)

    for param in uncertainty_model.parameters():
        param.requires_grad = False
    
    train_loss = 0
    correct, total = 0, 0
    real_label, fake_label = 0.5, 1
    train_loss_out, train_loss_in = 0, 0

    predictions_sigma = []
    average  = []
    variance = []
    predictions = []
    uncertainty = []
    peak_signal = []
    similarity  = []
    calibration = []
    meansquared = []
    
    for epoch in range(0, epochs):
        conductivity_model.train()
        prediction = conductivity_model(X_cond)
        predictions.append(prediction)
        conduction = prediction.permute(0, 2, 3, 1)
        mean, sigma = uncertainty_model(conduction)
        average.append(mean)
        variance.append(sigma)
        sigma = sigma.permute(0, 3, 1, 2)
        uncertainty.append(sigma)
        prediction = prediction + sigma 
        predictions_sigma.append(prediction)

        peaks, ssims, maces, rmsce = uncertainty_metrics(mean, sigma, prediction, y_cond)

        conduct_loss = torch.sqrt(loss_function(prediction, y_cond)) + maces

        peak_signal.append(peaks)
        similarity .append(ssims)
        calibration.append(maces)
        meansquared.append(rmsce)

        conductivity_optimizer.zero_grad()
        conduct_loss.backward()
        conductivity_optimizer.step()
        optimizer_F.zero_grad()
        uncertain_loss = loss_function(sigma, y_cond)
        uncertain_loss = uncertain_loss.to(torch.float32)
        utils.clip_grad_norm_(uncertainty_model.parameters(), max_norm = 2.0, norm_type = 2)
        optimizer_F.zero_grad()
        optimizer_F.step()
        train_loss += uncertain_loss.item()

        _, predicted = sigma.max(0)
        total += y_unc.size(0)
        correct += predicted.eq(y_cond).sum().item()
        label = torch.full((y_unc.shape[3], 1), real_label)
        optimizer_G.zero_grad()
        predict_in = uncertainty_model(X_unc, training_diffusion = True)
        predict_in = predict_in.to(torch.float32)

        loss_in = loss_function(predict_in, label).double().float()
        loss_in.requires_grad = True
        loss_in.backward()
        label.fill_(fake_label)

        inputs_out = 2 * torch.randn(1, m, m, 1) + X_unc
        predict_out = uncertainty_model(inputs_out, training_diffusion = True)
        loss_out = loss_function(predict_out, y_unc)
        loss_out.requires_grad = True
        loss_out.backward()
        train_loss_out += loss_out.item()
        train_loss_in += loss_in.item()
        optimizer_G.step()

        losses_condnet.append(conduct_loss), losses_sdenet.append(uncertain_loss/(len(X_unc)))
        if display:
            print(f'Epoch {epoch} \t CondNet-SDENet: {conduct_loss:.4f} \t PSNR: {np.abs(peaks.item()):.4f} \t SSIM: {np.abs(ssims.item()):.5f}')


    torch.save({ 
        'epoch':                epoch,
        'model_state_dict':     uncertainty_model.state_dict(),
        'optimizer_state_dict': optimizer_F.state_dict(),
        'loss':                 uncertain_loss/len(X_unc)
    }, 'sdenetcn.pth' )
    return get_output(peak_signal, similarity, calibration, losses_condnet, losses_sdenet, prediction, conduction, mean, sigma)


def sdenet_test(epochs, inputs, targets, depth = 6):
    conductivity_model, uncertainty_model = CondNet(), SDENetRegression(depth)
    conductivity_model.eval(), uncertainty_model.eval()
    conductivity_optimizer = optim.Adam(conductivity_model.parameters(), lr = 0.0001)
    losses_condnet, losses_sdenet = [], []
    test_loss = 0
    m, n = 128, 10
    transformA, transformB = transforms_init(m)
    num = np.random.choice(inputs.shape[0] - 5, 1)[0]
    loss_function = torch.nn.MSELoss()
    X, y = inputs[num:num+n, :, :, :], targets[num:num+n, :, :, :]
    X_cond, X_unc, y_unc, y_cond = transformA(X), transformA(X), transformB(y), transformB(y)
    X_unc, y_unc = X_unc.permute(0, 2, 3, 1), y_unc.permute(0, 2, 3, 1)

    with torch.no_grad():
        for epoch in range(0, epochs):
            current_mean = 0
            prediction = conductivity_model(X_cond)
            conduct_loss = loss_function(prediction, y_cond)
            conductivity_optimizer.zero_grad()
            conductivity_optimizer.step()

            prediction = prediction.permute(0, 2, 3, 1)
            for _ in range(10):
                mean, sigma = uncertainty_model(prediction)
                current_mean = current_mean + mean
            current_mean = current_mean / 10
            loss = loss_function(y_unc, sigma)
            test_loss += loss.item()
            losses_condnet.append(conduct_loss), losses_sdenet.append(np.sqrt(test_loss/epochs))
            print(f'Epoch {epoch} \tCondNet: {conduct_loss:.6f} | PSNR: {np.sqrt(test_loss/epochs):.4f}')
    return losses_condnet, losses_sdenet