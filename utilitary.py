import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

### Data import

def remove_empty_key(hf, keys):
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

### Preprocessing

# Edge detection

def find_contour(images, clamp_val=300, blur=5):

    """ Find the contour of all image in images
        Arguments:
            images : array of all images of interest.
        Return:
            all_contours : contains all the contours of all the images.
                           Has the same shape as images
            all_ret : contains all the threshold used for all the images.
                           Has the same shape as images.shape[0]
    """
    #single image
    if len(images.shape) == 2:
        images = images[np.newaxis,...]

    #pre-allocation
    #thresholds?
    all_ret = np.empty(images.shape[0])
    #contours as image
    contours_img = np.empty(images.shape)
    #contours as list
    contours_list = []

    for i, img in enumerate(images):

        img_c = img.copy()
        # Clamp the values of img from [0, max_val]
        img_c[img > clamp_val] = clamp_val
        # Change the type in uint8 for drawContours
        img_c = (img/clamp_val*255).astype(np.uint8)

        # Blur the image to remove noise
        # alternative : cv2.GaussianBlur(im,(3,3),0)
        img_c = cv2.medianBlur(img_c,blur)

        # Perform a binary threshold
        all_ret[i], th1 = cv2.threshold(img_c, 120, 255, cv2.THRESH_BINARY)
        # Find contours (return as a list)
        list_contour, hierarchy = cv2.findContours(th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Convert contours to image
        img_contour = cv2.drawContours(np.zeros(img_c.shape), list_contour, -1, (255,0,0), 3)
        # contour is in {0,1} <=> normalization
        img_contour /= 255

        contours_img[i,...] = img_contour
        contours_list.append(list_contour)

    return contours_img, all_ret, contours_list

# Max Intensity Projection

def single_MIP(data, list_keys, axis, channel=0):
    """ Max intensity projection (MIP) to axis
        Arguments :
            data : h5 data
            list_keys : list of keys contained in data
            axis : axis for the projection
            channel : channel to project, by default red
        Returns :
            MIP : np.array of dimensions (nb_vol, x, y, 3) of
                  the MIP.
                  The last dimension contains the RGB values
            MIP_avg : np.array of dimensions (nb_vol, x, y) of
                      the MIP.
                      The last dimension contains the mean RGB values
    """
    nb_vol = len(list_keys)
    x,y,z = data.get('0')['frame'][channel].shape

    #
    if axis == 2:
        MIP = np.empty((nb_vol, x, y))
    elif axis == 1:
        MIP = np.empty((nb_vol, x, z))
    else:
        MIP = np.empty((nb_vol, y, z))

    for i,key in enumerate(list_keys):
        MIP[i] = np.max(data.get(key)["frame"][channel], axis = axis)

    return MIP

def MIP_GR(data, list_keys, axis):
    """ Max intensity projection (MIP) of red and green channels separately to axis
        Arguments :
            data : h5 data
            list_keys : list of keys contained in data
            axis : axis for the projection
        Returns :
            rMIP : np.array of dimensions (nb_vol, x, y, 3) of
                  the MIP of the red channel
            gMIP : np.array of dimensions (nb_vol, x, y, 3) of
                  the MIP of the green channel
    """
    # project red channel along the axis
    r_MIP = single_MIP(data, list_keys, axis, 0)

    # project green channel along the axis
    g_MIP = single_MIP(data, list_keys, axis, 1)

    return r_MIP, g_MIP

def np_MIP(array, list_keys, axis):
    """ Max intensity projection (MIP) to axis
        Arguments :
            data : numpy array
            list_keys : list of keys contained in data
            axis : axis for the projection
        Returns :
            MIP : np.array of dimensions (nb_vol, x, y, 3) of
                  the MIP.
    """
    nb_vol, x,y,z = array.shape

    if axis == 2:
        MIP = np.empty((nb_vol, x, y))
    elif axis == 1:
        MIP = np.empty((nb_vol, x, z))
    else:
        MIP = np.empty((nb_vol, y, z))

    for i,key in enumerate(list_keys):
        MIP[i] = np.max(array[i], axis = axis)

    return MIP

# Cropping

def crop_ctr_mass(img, size=128, nb_contour = 2):
    """ Crop image around its contours center of mass
        Arguments :
            img : numpy array
            size : distance to crop for each direction
        Returns :
            _ : np.array of dimensions (2*size,2*size)
        Use minimum mode to pad if the cropping exceed the
        initial image dimensions
    """

    #Find the contour of img
    _, _, contours_list = find_contour(img, clamp_val=300, blur=5)

    #Compute perimeter of each closed contour
    perim_contours = [(i, cv2.arcLength(cnt, True)) for i, cnt in enumerate(contours_list[0])]
    #Sort contour by perimeter size
    perim_contours.sort(key=lambda tup: tup[1], reverse=True)

    tmp = np.zeros((img.shape))

    # Draw the two biggest closed contour
    for contour in perim_contours[:nb_contour]:
        tmp = cv2.drawContours(tmp, contours_list[0][contour[0]], -1, (255,0,0), 3)

    #compute the center of mass of the binary contours
    (x, y) = ndimage.center_of_mass(tmp)

    #convert to int
    (x, y) = (int(x),int(y))

    #the cropping exceed the image dimensions
    if x-size < 0 or x+size > img.shape[0] or \
        y-size < 0 or y+size > img.shape[1]:
        #padding
        img_pad=np.pad(img, size, mode='minimum')
        return img_pad[x:x+2*size,y:y+2*size]
    #the cropping in within the image dimensions
    else:
        return img[x-size:x+size,y-size:y+size]

def crop_ctr_mass_GR(img_r, img_g, img_r_ctr, size=128):
    """ Crop image around its contours center of mass
        Arguments :
            img_r : numpy array, red channel
            img_g : numpy array, green channel
            img_ctr : numpy array version of img_r contours
            size : distance to crop for each direction
        Returns :
            _ : np.array of dimensions (2*size,2*size)
        Use minimum mode to pad if the cropping exceed the
        initial image dimensions
    """
    cropped_red = np.empty((len(img_r), 2*size, 2*size))
    cropped_green = np.empty((len(img_g), 2*size, 2*size))
    cropped_contours = np.empty((len(img_r), 2*size, 2*size))

    #iterate over images
    for i, im in enumerate(img_r):
        #obtian contours
        img_ctr = img_r_ctr[i]
        #compute contours center of mass
        (x, y) = ndimage.center_of_mass(img_ctr)
        (x, y) = (int(x), int(y))

        #cropping exceed image border
        if x-size < 0 or x+size > im.shape[0] or \
            y-size < 0 or y+size > im.shape[1]:

            img_pad = np.pad(im, size, mode='minimum')
            img_pad2 = np.pad(img_g[i], size, mode='minimum')
            img_pad3 = np.pad(img_r_ctr[i], size, mode='minimum')

            cropped_red[i] = img_pad[x:x+2*size,y:y+2*size]
            cropped_green[i] = img_pad2[x:x+2*size,y:y+2*size]
            cropped_contours[i] = img_pad3[x:x+2*size,y:y+2*size]
        else:
            cropped_red[i] = im[x-size:x+size,y-size:y+size]
            cropped_green[i] = img_g[i, x-size:x+size,y-size:y+size]
            cropped_contours[i] = img_ctr_all[i, x-size:x+size,y-size:y+size]

    return cropped_red, cropped_green, cropped_contours

# Segmentation
def rectangle_neurons(img, contours_list, nb_contour=2):
    """ Return the coordinates of a bounding rectangle for each
        contour (in descending area order)
        Arguments :
            img : numpy array
            contours_list : img contours in list format
            nb_contour : number of rectables returned
        Returns :
            rect_neurons : x, y, h, for each rectangle, (x,y) is
            the position of the top left corner
    """
    #compute area of each contour
    area_contours = [[cv2.contourArea(i), i] for i in contours_list]

    # sort contour by area
    area_contours.sort(key=lambda x : x[0], reverse=True)

    #list of rectangle position bounding each contour
    rect_neurons = []

    # use the biggest two contours
    for contour in area_contours[:nb_contour]:
        (x, y, w, h) = cv2.boundingRect(contour[1])

        #increase the boundaries
        h += 10
        w += 10
        x -= 5
        y -= 5
        rect_neurons.append([x,y,w,h])

    return rect_neurons

# Rotation
def rot_img(img, img_ctr, list_ctr):
    """ Rotate the image so that the line passing throught the 2
    biggest contour center of mass is horitonzal
        Arguments :
            img : numpy array
            img_ctr : img contours in img format
            list_ctr : img contours in list format
        Returns :
            _ : rotated image
    """
    #bound the two biggest contour with a rectangle
    rect_neurons = rectangle_neurons(img, list_ctr,2)
    ctr_mass_list = []

    #iterate on the bounding rectangle
    for i in rect_neurons:
        y, x, w, h = i
        #padding to conserve the coordinates -> might not be necessary,
        #could probably work with just img_ctr[x:x+w,y:y+h]
        tmp_img = np.zeros(img.shape)
        tmp_img[x:x+w,y:y+h] = img_ctr[x:x+w,y:y+h]
        (x, y) = ndimage.center_of_mass(tmp_img)
        ctr_mass_list.append((int(x),int(y)))

    #compute the rotation angle
    dx = ctr_mass_list[0][0] - ctr_mass_list[1][0]
    dy = ctr_mass_list[0][1] - ctr_mass_list[1][1]
    angle = np.degrees(np.arctan(dx/dy))

    return ndimage.rotate(img, angle)


### Voxelmorph set generation

def vxm_data_generator(x_data, idx_fixed=None, vol_fixed=[], batch_size=32):
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
        if len(vol_fixed) != 0:
            #dummy indexes
            idx2 = np.full((batch_size),0)
            tmp = np.array([vol_fixed])
            fixed_images = tmp[idx2,...,np.newaxis]
        else:
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

def create_xy(val_data, fixed_idx):
    """ Create a validation set for model.fit, of the same shape as
        vxm_data_generator, but that returns the data as tuple
        Arguments :
            val_data : data for validation
            fixed_idx : index of the fixed (reference) image
        Returns :
            tuple (x,y)
            x = [moving_data, fixed_data]
            y = [fixed_data, zero_phi]

    """
    data = val_data[:-1]
    ndims = len(data.shape[1:])
    fixed_slice = val_data[fixed_idx,...]
    fixed_data = (np.ones(data.shape) * fixed_slice)[..., np.newaxis]
    moving_data = data[..., np.newaxis]
    zero_phi = np.zeros([*data.shape, ndims])
    x = [moving_data, fixed_data]
    y = [fixed_data, zero_phi]
    return (x,y)

def create_xy_3d(slices, fixed_vol):

    nb_samples = len(slices)
    idx2 = np.full((nb_samples),0)
    tmp = np.array([fixed_vol])

    fixed_data = tmp[idx2,...,np.newaxis]

    moving_data = slices[..., np.newaxis]

    zero_phi = np.zeros([*slices.shape, nb_samples])
    x = [moving_data, fixed_data]
    y = [fixed_data, zero_phi]

    return (x,y)

### Helper

def plot_history(hist, param, loss_name=['loss','val_loss'], save_name = 'title'):
    # Simple function to plot training history.
    if len(loss_name)== 2:
        plt.figure()
        plt.plot(hist.epoch, hist.history[loss_name[0]], '.-')
        plt.plot(hist.epoch, hist.history[loss_name[1]], '.-')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend([loss_name[0], loss_name[1]])
        plt.title(str(param))
        plt.show()
    elif len(loss_name)== 1:
        plt.figure()
        plt.plot(hist.epoch, hist.history[loss_name], '.-')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title(str(param))
        plt.show()

def export_history(hist, filename):
    with open(filename,'w') as trg_file:
        for epoch, loss, val_loss in zip(hist.epoch, hist.history['loss'], hist.history['val_loss']):
            trg_file.write(str(epoch)+'\t'+str(loss)+'\t'+str(val_loss)+'\n')
