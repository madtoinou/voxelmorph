import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

def vxm_data_generator(x_data, idx_fixed=None, batch_size=32):
    """
    Generator that takes in data of size [N, H, W], and yields data for
    our custom vxm model. Note that we need to provide numpy data for each
    input, and each output.

    inputs:  moving [bs, H, W, 1], fixed image [bs, H, W, 1]
    outputs: moved image [bs, H, W, 1], zero-gradient [bs, H, W, 2]
    """

    # preliminary sizing
    vol_shape = x_data.shape[1:] # extract data shape
    ndims = len(vol_shape)

    # prepare a zero array the size of the deformation
    # we'll explain this below
    zero_phi = np.zeros([batch_size, *vol_shape, ndims])

    while True:
        # prepare inputs:
        # images need to be of the size [batch_size, H, W, 1]
        idx1 = np.random.randint(0, x_data.shape[0], size=batch_size)
        moving_images = x_data[idx1, ..., np.newaxis]
        if idx_fixed == None:
            idx2 = np.random.randint(0, x_data.shape[0], size=batch_size)
        else:
            idx2 = np.full((batch_size),idx_fixed)
        fixed_images = x_data[idx2, ..., np.newaxis]
        inputs = [moving_images, fixed_images]

        # prepare outputs (the 'true' moved image):
        # of course, we don't have this, but we know we want to compare
        # the resulting moved image with the fixed image.
        # we also wish to penalize the deformation field.
        outputs = [fixed_images, zero_phi]

        yield (inputs, outputs)

def plot_history(hist, loss_name='loss', save_name = 'title'):
    # Simple function to plot training history.
    plt.figure()
    plt.plot(hist.epoch, hist.history[loss_name], '.-')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title(save_name)
    title = 'Hist' + save_name + ".pdf"
    plt.savefig(title)
    plt.show()

def remove_empty_key(keys, hf):
    """ Remove the empty keys in the list keys
        Arguments :
            keys : list of all the keys
        Return :
            use_key : all the keys that are not empty
    """
    use_key = []
    for key in keys:
        if len(hf.get(key))!= 0:
            use_key.append(key)
    return use_key

def to_MIP(data, list_keys, axis):
    """ Mean intensity projection (MIP) to axis
        Arguments :
            data : h5 data
            list_keys : list of keys contained in data
            axis : axis for the projection
        Returns :
            MIP : np.array of dimensions (nb_vol, x, y, 3) of
                  the MIP.
                  The last dimension contains the RGB values
            MIP_avg : np.array of dimensions (nb_vol, x, y) of
                      the MIP.
                      The last dimension contains the mean RGB values
    """
    nb_vol = len(list_keys)
    x,y,z = data.get('0')['frame'][0].shape

    #
    MIP = np.empty((nb_vol, x, y, 3))
    MIP_avg = np.empty((nb_vol, x, y))
    MIP_3 = np.zeros((x, y))

    for i,key in enumerate(list_keys):
        MIP_1 = np.max(data.get(key)["frame"][0], axis = axis)
        MIP_2 = np.max(data.get(key)["frame"][1], axis = axis)
        MIP[i, ...] = np.dstack((MIP_1/255, MIP_2/255, MIP_3/255))
        MIP_avg[i, ...] = (MIP_1/255 + MIP_2/255 + MIP_3/255)/3

    return MIP, MIP_avg

def find_contour(images, mod=1.1, clamp_val=300, blur=5):

    """ Find the contour of all image in images
        Arguments:
            images : array of all images of interest. The range of the intensity
                     values of the images should be divided by 255 before passing
                     it in argument
        Return:
            all_contours : contains all the contours of all the images.
                           Has the same shape as images
            all_ret : contains all the threshold used for all the images.
                           Has the same shape as images.shape[0]
    """
    if len(images.shape) == 2:
        images = images[np.newaxis,...]
    all_contours = np.empty(images.shape)
    all_ret = np.empty(images.shape[0])

    for i, im in enumerate(images):

        # Get the original values of the image back
        if np.max(im) < 255 :
            im2 = im * 255
        # The image is already in its original form
        else :
            im2 = im
        # Clamp the values of im2 from [0, max_val], because some intensity
        # values are > 255
        max_val = clamp_val
        im2[im2 > max_val] = max_val
        # Change the type in uint8 for drawContours
        im2 = (im2/max_val*255).astype(np.uint8)

        # Blur the image to remove noise
        # You can also use : cv2.GaussianBlur(im,(3,3),0)
        im2 = cv2.medianBlur(im2,blur)

        # Perform a binary threshold + OTSU
        all_ret[i], th1 = cv2.threshold(im2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Use a threshold = optimal threshold * 1.1 = all_ret[i]*1.1
        # You can also use cv2.adaptiveThreshold(im2,im2.max(),cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        # cv2.THRESH_BINARY,3,2)
        _, th1 = cv2.threshold(im2, all_ret[i]*mod, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = cv2.drawContours(np.zeros(im.shape), contours, -1, (255,0,0), 3)
        # contour is in {0,1}
        contour /= 255
        all_contours[i,...] = contour

    return all_contours, all_ret

def remove_empty_key(hf, keys):
    use_key = []
    for key in keys:
        if len(hf.get(key))!= 0:
            use_key.append(key)
    return use_key

def red_MIP(data, list_keys, axis):
    """ Mean intensity projection (MIP) to axis 
        Arguments :
            data : h5 data
            list_keys : list of keys contained in data
            axis : axis for the projection
        Returns :
            MIP : np.array of dimensions (nb_vol, x, y, 3) of 
                  the MIP.
                  The last dimension contains the RGB values
            MIP_avg : np.array of dimensions (nb_vol, x, y) of 
                      the MIP.
                      The last dimension contains the mean RGB values
    """
    nb_vol = len(list_keys)
    x,y,z = data.get('0')['frame'][0].shape

    #
    if axis == 2:
        r_MIP = np.empty((nb_vol, x, y))
    elif axis == 1:
        r_MIP = np.empty((nb_vol, x, z))
    else:
        r_MIP = np.empty((nb_vol, y, z))

    for i,key in enumerate(list_keys):
        r_MIP[i] = np.max(data.get(key)["frame"][0], axis = axis)
            
    return r_MIP

def crop_ctr_mass(img, size=150):
    img_ctr, _ = find_contour(img,0.8,blur=5)
    img_ctr = img_ctr.reshape(img.shape)
    (x, y) = ndimage.center_of_mass(img_ctr)
    (x,y) = (int(x),int(y))
    if x-size < 0 or x+size > img.shape[0] or \
        y-size < 0 or y+size > img.shape[1]:
        img_pad=np.pad(img,size,mode='minimum')
        return img_pad[x:x+2*size,y:y+2*size]
    else:
        return img[x-size:x+size,y-size:y+size]

def Sobel_edges(MIPs):
    #edge detection
    mask = np.array([[1, 0, -1],
                     [2, 0, -2],
                     [1, 0, -1]])
    MIPs_sobel = np.empty(MIPs.shape)

    for i, img in enumerate(MIPs):

        MIPs_sobel[i] = ndi.convolve(ndi.convolve(img, mask, output=np.int16), mask.T, output=np.int16)
    
    return MIPs_sobel