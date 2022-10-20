import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk



def sigmoid_mapping(image, curve_steepness = 0.01, output_min = 0, output_max = 1.0, intensity_midpoint = None):
    '''
    Map the image using a sigmoid function.
    Args:
        image (SimpleITK image): scalar input image.
        curve_steepness: Control the sigmoid steepness, the larger the number the steeper the curve.
        output_min: Minimum value for output image, default 0.0 .
        output_max: Maximum value for output image, default 1.0 .
        intensity_midpoint: intensity value defining the sigmoid midpoint (x coordinate), default is the
                            median image intensity.
    Return:
        SimpleITK image with float pixel type.
    '''
    if intensity_midpoint is None:
        intensity_midpoint = np.median(sitk.GetArrayViewFromImage(image))

    sig_filter = sitk.SigmoidImageFilter()
    sig_filter.SetOutputMinimum(output_min)
    sig_filter.SetOutputMaximum(output_max)
    sig_filter.SetAlpha(1.0/curve_steepness)
    sig_filter.SetBeta(float(intensity_midpoint))
    return sitk.GetArrayFromImage(sig_filter.Execute(sitk.Cast(image, sitk.sitkFloat64)))

def myshow(img, title=None, margin=0.05, dpi=80):
    nda = sitk.GetArrayViewFromImage(img)
    spacing = img.GetSpacing()    
    if nda.ndim == 3:
        c = nda.shape[-1]
        if not c in (3,4):
            nda = nda[nda.shape[0]//2,:,:]
    
    elif nda.ndim == 4:
        c = nda.shape[-1]
        if not c in (3,4):
            pass
            # raise Runtime("Unable to show 3D-vector Image")
            
        nda = nda[nda.shape[0]//2,:,:,:]
            
    ysize = nda.shape[0]
    xsize = nda.shape[1]
      
    figsize = (20, 10)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2*margin])
    extent = (0, xsize*spacing[1], ysize*spacing[0], 0)
    t = ax.imshow(nda,extent=extent,interpolation=None)
    if nda.ndim == 2:
        t.set_cmap("gray")
    if(title):
        plt.title(title)

def myshow3d(img, xslices = [], yslices = [], zslices = [], title = None, margin = 0.05, dpi = 80):
    '''
    myshow3d(t1w, yslices = range(0, t1w.GetSize()[1]-50, 20), zslices=range(50, t1w.GetSize()[2]-50, 20), dpi = 100)
    '''
    size = img.GetSize()
    img_xslices = [img[s,:,:] for s in xslices]
    img_yslices = [img[:,s,:] for s in yslices]
    img_zslices = [img[:,:,s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))   
    img_null = sitk.Image([0,0], img.GetPixelID(), img.GetNumberOfComponentsPerPixel()) 
    img_slices = []
    d = 0
    if len(img_xslices):
        img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
        d += 1 
    if len(img_yslices):
        img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
        d += 1 
    if len(img_zslices):
        img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
        d +=1
    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen,d])
        else:
            img_comps = []
            for i in range(0,img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen,d]))
            img = sitk.Compose(img_comps)
              
    myshow(img, title, margin, dpi)