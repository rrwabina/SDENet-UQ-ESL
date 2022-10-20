import SimpleITK as sitk
import matplotlib.pyplot as plt


def bias_field_correction(inputImage):
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)
    maskImage = sitk.OtsuThreshold(inputImage)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    correctedImage = corrector.Execute(inputImage, maskImage)
    log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)
    bias_field = inputImage / sitk.Exp(log_bias_field)
    return maskImage, correctedImage, log_bias_field

def bfc_plot(inputImage, maskImage, correctedImage, log_bias_field, mriType: str):
    nda_orig = sitk.GetArrayFromImage(inputImage)
    nda_mask = sitk.GetArrayFromImage(maskImage)
    nda_corr = sitk.GetArrayFromImage(correctedImage)
    nda_bias = sitk.GetArrayFromImage(log_bias_field)
    
    fig = plt.figure(figsize = (15, 10))
    if mriType == 't1w':
        fig.add_subplot(1, 4, 1)
        plt.imshow(nda_orig[:, :, 100], cmap = 'inferno')
        plt.title('Original ' + mriType)
        
        fig.add_subplot(1, 4, 2)
        plt.imshow(nda_mask[:, :, 100], cmap = 'inferno')
        plt.title('Mask ' + mriType)

        fig.add_subplot(1, 4, 3)
        plt.imshow(nda_bias[:, :, 100], cmap = 'inferno')
        plt.title('Bias ' + mriType)

        fig.add_subplot(1, 4, 4)
        plt.imshow(nda_corr[:, :, 100], cmap = 'inferno')
        plt.title('BFC ' + mriType)
    elif mriType == 't2w' or mriType == 't2s':
        fig.add_subplot(1, 4, 1)
        plt.imshow(nda_orig[30, :, :], cmap = 'inferno')
        plt.title('Original ' + mriType)
        
        fig.add_subplot(1, 4, 2)
        plt.imshow(nda_mask[30, :, :], cmap = 'inferno')
        plt.title('Mask ' + mriType)

        fig.add_subplot(1, 4, 3)
        plt.imshow(nda_bias[30, :, :], cmap = 'inferno')
        plt.title('Bias ' + mriType)

        fig.add_subplot(1, 4, 4)
        plt.imshow(nda_corr[30, :, :], cmap = 'inferno')
        plt.title('BFC ' + mriType)