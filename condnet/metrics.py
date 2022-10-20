import tensorflow as tf
import torch

def _binarize(y_data, threshold):
    """
    args:
        y_data : [float] 4-d tensor in [batch_size, channels, img_rows, img_cols]
        threshold : [float] [0.0, 1.0]
    return 4-d binarized y_data
    """
    y_data[y_data < threshold]  = 0.0
    y_data[y_data >= threshold] = 1.0
    return y_data
    
class PSNR(object):
    def __init__(self, des = "Peak Signal to Noise Ratio"):
        self.des = des

    def __repr__(self):
        return "PSNR"

    def __call__(self, y_pred, y_true, dim = 1, threshold=None):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            threshold : [0.0, 1.0]
        return PSNR, larger the better
        """
        if threshold:
            y_pred = _binarize(y_pred, threshold)
        mse = torch.mean((y_pred - y_true) ** 2)
        return 10 * torch.log10(1 / mse)

def mean_squared_error(output, target, is_mean=False):
    output, target = output.detach().numpy(), target.detach().numpy()
    if is_mean:
        mse = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(output, target), [1, 2, 3]))
    else:
        mse = tf.reduce_mean(tf.reduce_sum(tf.math.squared_difference(output, target), [1, 2, 3]))
    return mse 

def accuracy(prediction, targets):
    rounded_prediction = torch.round(torch.sigmoid(prediction))
    correct = (prediction == targets[0 : 3249]).float()
    return (correct.sum()/len(correct)).item()

def dice_coeff(prediction, target):
    prediction = prediction[9:10, :, :, :]
    smooth = 1.
    num = prediction.size(0)
    m1 = prediction.view(num, -1).float()      
    m2 = target.view(num, -1).float()  
    intersection = (m1 * m2).sum().float()
    return ((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)).item()

