import SimpleITK as sitk
from IPython.display import clear_output

import matplotlib.pyplot as plt
import os

def image_smooth(mri_image):
    return sitk.CurvatureFlow(image1 = mri_image, timeStep = 0.125, numberOfIterations = 3)

def start_plot():
    global metric_values, multires_iterations
    metric_values = []
    multires_iterations = []

def end_plot():
    global metric_values, multires_iterations
    del metric_values
    del multires_iterations
    plt.close()

def update_multires_iterations():
    global metric_values, multires_iterations
    multires_iterations.append(len(metric_values))

def plot_values(registration_method):
    global metric_values, multires_iterations

    metric_values.append(registration_method.GetMetricValue())
    clear_output(wait=True)
    plt.plot(metric_values, "r")
    plt.plot(
        multires_iterations,
        [metric_values[index] for index in multires_iterations],
        "b*")
    plt.xlabel("Iteration Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.show()

def initial_alignment(fixed_image, moving_image):
    initial_transform = sitk.CenteredTransformInitializer(fixed_image, moving_image, 
                                sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)
    moving_resampled = sitk.Resample(fixed_image, moving_image, initial_transform, 
                                sitk.sitkLinear, 0.0, moving_image.GetPixelID())                        
    return initial_transform, moving_resampled

def registration(fixed_image, moving_image, initial_transform, show_plot = False):
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate = 0.01, numberOfIterations = 100, 
        convergenceMinimumValue = 1e-6,
        convergenceWindowSize = 10)
    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas = [2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInitialTransform(initial_transform, inPlace = False)

    if show_plot == True:
        registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations)
        registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
    metrics = registration_method.GetMetricValue()
    stop_condition = registration_method.GetOptimizerStopConditionDescription()

    final_transform = registration_method.Execute(sitk.Cast(fixed_image, sitk.sitkFloat32), sitk.Cast(moving_image, sitk.sitkFloat32))
    moving_resampled = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
    return final_transform, moving_resampled

def save_registration(final_transform, moving_resampled, key):
    directory = 'preprocess'
    sitk.WriteImage(moving_resampled, os.path.join(directory, str(key) + "_resampled.nii.gz"))
    #sitk.WriteTransform(final_transform, os.path.join(directory, "final_transform.nii.gz"))