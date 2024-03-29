import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import uncertainty_toolbox as uct

def _binarize(y_data, threshold):
    y_data[y_data < threshold]  = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data

class PSNR(object):
    def __init__(self, des = 'Peak Signal to Noise Ratio'):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim = 1, threshold = None):
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1 / mse)

class SSIM(object):
    def __init__(self, des = 'structural similarity index'):
        self.des = des

    def __repr__(self):
        return "SSIM"

    def gaussian(self, w_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - w_size//2)**2/float(2*sigma**2)) for x in range(w_size)])
        return gauss/gauss.sum()

    def create_window(self, w_size, channel=1):
        _1D_window = self.gaussian(w_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, w_size, w_size).contiguous()
        return window

    def __call__(self, y_pred, y_true, w_size = 11, size_average = True, full = False):
        if torch.max(y_pred) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(y_pred) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val

        padd = 0
        (_, channel, height, width) = y_pred.size()
        window = self.create_window(w_size, channel = channel).to(y_pred.device)

        mu1 = F.conv2d(y_pred, window, padding = padd, groups = channel)
        mu2 = F.conv2d(y_true, window, padding = padd, groups = channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(y_pred * y_pred, window, padding = padd, groups = channel) - mu1_sq
        sigma2_sq = F.conv2d(y_true * y_true, window, padding = padd, groups = channel) - mu2_sq
        sigma12   = F.conv2d(y_pred * y_true, window, padding = padd, groups = channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret

def brier_score(y_pred, y_true):
  return 1 + (np.sum(y_pred ** 2) - 2 * np.sum(y_pred[np.arange(y_pred.shape[0]), y_true])) / y_true.shape[0]

def uncertainty_metrics(mean, sigma, prediction, target, calibration_error = True):
    psnr, ssim = PSNR(), SSIM()

    peaks = psnr(prediction, target)
    ssims = ssim(prediction, target)
    if calibration_error:
        maces = uct.mean_absolute_calibration_error(    mean[:, 0, 0].detach().numpy(), sigma[:, 0, 0, 0].detach().numpy(), target[:, 0, 0, 0].detach().numpy(), recal_model = None)
        rmsce = uct.root_mean_squared_calibration_error(mean[:, 0, 0].detach().numpy(), sigma[:, 0, 0, 0].detach().numpy(), target[:, 0, 0, 0].detach().numpy(), recal_model = None)
        return peaks, ssims, maces, rmsce
    else:
        return peaks, ssims