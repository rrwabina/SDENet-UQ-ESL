import torch.nn as nn
import numpy as np
import nibabel as nib
from IPython.display import 


def standardize(image):
    standardized_image = np.zeros(image.shape)
    centered_scaled = 0
    for c in range(image.shape[0]):
        for z in range(image.shape[3]):
            image_slice = image[c,:,:,z]

            centered = image_slice - np.mean(image_slice)
            if np.std(centered) != 0:
                centered_scaled = centered / np.std(centered)
            # update  the slice of standardized image
            # with the scaled centered and scaled image
            standardized_image[c, :, :, z] = centered_scaled
    return standardized_image


from IPython.display import clear_output
def slice_viewer_generator(dim, img, img_label):
    img_switcher = {
        0: lambda i: img[i,:,:],
        1: lambda i: img[:,i,:],
        2: lambda i: img[:,:,i]
    }
    img_label_switcher = {
        0: lambda i: img_label[i,:,:],
        1: lambda i: img_label[:,i,:],
        2: lambda i: img_label[:,:,i]
    }
    
    for s in range(img.shape[dim]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
        ax1.set_title('Image')
        ax2.set_title('Label')
        fig.suptitle(f'Viewing slice {s} along dimension {dim}')
        ax1.imshow(img_switcher[dim](s), cmap="inferno")
        ax2.imshow(img_label_switcher[dim](s), cmap="inferno")
        ax1.set_title('Label')
        plt.show()
        clear_output(wait=True)
        yield

def display_volume_slices(img, w, h):
    plot_w = w
    plot_h = h

    # Use figsize parameter to adjust image size
    fig, ax = plt.subplots(plot_h, plot_w, figsize=[35,35])

    for i in range(plot_w*plot_h):
        plt_x = i % plot_w
        plt_y = i // plot_w
        if (i < len(img)):
            ax[plt_y, plt_x].set_title(f"slice {i}")
            ax[plt_y, plt_x].imshow(img[i], cmap='inferno')
        ax[plt_y, plt_x].axis("off")

    plt.show()

def vol_gen():
    filenames = os.listdir(f"../dataset/images/train")
    for filename in filenames:
        vol = nib.load(f"../dataset/images/train/{filename}").get_fdata()
        vr = np.zeros((vol.shape[1], vol.shape[2]))
        for z in range (vol.shape[0]):
            vr += vol[z,:,:]

        # Place inside the loop for animation!
        plt.imshow(vr, cmap="gray")
        plt.title(f"{filename}") 
        plt.show()
        clear_output(wait=True)
        yield


