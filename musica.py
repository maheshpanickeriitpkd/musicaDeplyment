# Script for running MUSICA algorithm on a grayscale image:
# Written by Lafith Mattara on 2021-06-05


#import logging
import numpy as np
import copy
from skimage.transform import pyramid_reduce, pyramid_expand
from skimage.transform import resize
# from skimage import io, img_as_float
# import cv2
#import matplotlib.pyplot as plt


# logging.getLogger('matplotlib').setLevel(logging.WARNING)
# logger = logging.getLogger(__name__)
#change level here for console output
# logger.setLevel(logging.INFO)

# file_handler = logging.FileHandler('musica_py.log')
# formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s')
# file_handler.setFormatter(formatter)
# logger.addHandler(file_handler)

# stream_handler = logging.StreamHandler()
# stream_format = logging.Formatter('%(levelname)s:\t%(message)s')
# stream_handler.setFormatter(stream_format)
# logger.addHandler(stream_handler)


def non_linear_gamma_correction(img, params):
    """Non linear gamma correction

    Parameters
    ----------    
    img : Image        
    params : dict
        Store values of a, M and p.

    Returns
    -------
    en_img: Enhanced Image
        
    """
    # Non linear operation goes here:
    M = params['M']
    p = params['p']
    a = params['a']           
    
    en_img = a*M*np.multiply(
        np.divide(
            img, np.abs(img), out=np.zeros_like(img),where=img != 0),
        np.power(
            np.divide(
                np.abs(img), M), p))
    return en_img


def display_pyramid(pyramid):
    """Function for plotting all levels of an image pyramid

    Parameters
    ----------
    pyramid : list
        list containing all levels of the pyramid
    """
    rows, cols = pyramid[0].shape
    composite_image = np.zeros((rows, cols + (cols // 2)), dtype=np.double)
    composite_image[:rows, :cols] = pyramid[0]
    i_row = 0
    for p in pyramid[1:]:
        n_rows, n_cols = p.shape[:2]
        composite_image[i_row:i_row + n_rows, cols:cols + n_cols] = p
        i_row += n_rows
    fig, ax = plt.subplots()
    ax.imshow(composite_image, cmap='gray')
    plt.show()


def isPowerofTwo(x):
    # check if number x is a power of two
    return x and (not(x & (x - 1)))


def findNextPowerOf2(n):
    # taken from https://www.techiedelight.com/round-next-highest-power-2/
    # Function will find next power of 2

    # decrement `n` (to handle cases when `n` itself
    # is a power of 2)
    n = n - 1
    # do till only one bit is left
    while n & n - 1:
        n = n & n - 1  # unset rightmost bit
    # `n` is now a power of two (less than `n`)
    # return next power of 2
    return n << 1


def resize_image(img):
    """MUSICA works for dimension like 2^N*2^M.
    Hence padding is required for arbitrary shapes

    Parameters
    ----------
    img : numpy.ndarray
        Original image

    Returns
    -------
    numpy.ndarray
        Resized image after padding
    """
    row, col = img.shape
    # check if dimensions are power of two
    # if not pad the image accordingly
    # logger.debug("Calculating how much padding is required...")
    if isPowerofTwo(row):
        rowdiff = 0
    else:
        nextpower = findNextPowerOf2(row)
        rowdiff = nextpower - row

    if isPowerofTwo(col):
        coldiff = 0
    else:
        nextpower = findNextPowerOf2(col)
        coldiff = nextpower - col

    img_ = np.pad(
        img,
        ((0, rowdiff), (0, coldiff)),
        'reflect')
    # logger.info(
    #         'Image padded from [{},{}] to [{},{}]'.format(
    #             img.shape[0], img.shape[1],
    #             img_.shape[0],img_.shape[1]))
    return img_


def gaussian_pyramid(img, L):
    """Function for creating a Gaussian Pyramid

    Parameters
    ----------
    img : numpy.ndarray
        Input image or g0.
    L : Int
        Maximum level of decomposition.

    Returns
    -------
    list
        list containing images from g0 to gL in order
    """
    # logger.debug('Creating Gaussian pyramid...')
    # Gaussian Pyramid
    tmp = copy.deepcopy(img)
    gp = [tmp]
    for layer in range(L):
        # logger.debug('creating Layer %d...' % (layer+1))
        tmp = pyramid_reduce(tmp, preserve_range=True)
        gp.append(tmp)
    # logger.info('Finished creating Gaussian Pyramid')
    return gp


def laplacian_pyramid(img, L):
    """Function for creating Laplacian Pyramid

    Parameters
    ----------
    img : numpy.ndarray
        Input image or g0.
    L : Int
        Max layer of decomposition

    Returns
    -------
    list
        list containing laplacian layers from L_0 to L_L in order
    list
        list containing layers of gauss pyramid
    """
    gauss = gaussian_pyramid(img, L)
    # logger.debug('Creating Laplacian pyramid...')
    # Laplacian Pyramid:
    lp = []
    for layer in range(L):
        # logger.debug('Creating layer %d' % (layer))
        tmp = pyramid_expand(gauss[layer+1], preserve_range=True)
        tmp = gauss[layer] - tmp
        lp.append(tmp)
    lp.append(gauss[L])
    # logger.info("Finished creating Laplacian pyramid")
    return lp, gauss


def enhance_coefficients(laplacian, L, params):
    """Non linear operation of pyramid coefficients

    Parameters
    ----------
    laplacian : list
        Laplacian pyramid of the image.
    L : Int
        Max layer of decomposition
    params : dict
        Store values of a, M and p.

    Returns
    -------
    list
        List of enhanced pyramid coeffiencts.
    """
    # logger.debug('Non linear transformation of coefficients...')
    # Non linear operation goes here:
    M = params['M']
    p = params['p']
    a = params['a']
    xc = params['xc']
    for layer in range(L):
        # logger.info('Modifying Layer %d' % (layer))
        x = laplacian[layer]
        # x[x < 0] = 0.0  # removing all negative coefficients
        G = a[layer]*M
        
        # xc = 0.0002*M
        x_x = np.divide(
                x, np.abs(x),
                out=np.zeros_like(x),
                where=x != 0)
        x_x = np.divide(
                x, xc,
                out=x_x,
                where=np.abs(x)<xc)
        x_mp = np.power(
                np.divide(
                    np.abs(x), M), p[layer])
        x_mp = np.power(
                np.divide(
                    xc, M,
                    out=x_mp,
                    where=np.abs(x)<xc),
                p[layer])

        laplacian[layer] = G*np.multiply(x_x,x_mp)
    return laplacian


def reconstruct_image(laplacian, L):
    """Function for reconstructing original image
    from a laplacian pyramid

    Parameters
    ----------
    laplacian : list
        Laplacian pyramid with enhanced coefficients
    L : int
        Max level of decomposition

    Returns
    -------
    numpy.ndarray
        Resultant image matrix after reconstruction.
    """
    # logger.debug('Reconstructing image...')
    # Reconstructing original image from laplacian pyramid
    rs = laplacian[L]
    for i in range(L-1, -1, -1): 
        rs = pyramid_expand(rs, preserve_range=True)
        rs = np.add(rs, laplacian[i])
        # logger.debug('Layer %d completed' % (i))
    # logger.info(
    #         'Finished reconstructing image from modified pyramid')
    return rs


def musica(img, L, params, debug=True):
    """Function for running MUSICA algorithm

    Parameters
    ----------
    img : numpy.ndarray
        Input image
    L : int
        Max level of decomposition
    params : dict
        Contains parameter values required
        for non linear enhancement
    plot : bool, optional
        To plot the result, by default False

    Returns
    -------
    numpy.ndarray
        Final enhanced image with original dimensions
    """
    # if debug is True:
    #     logger.setLevel(logging.INFO)
    # else:
    #     logger.setLevel(logging.CRITICAL)
        
    nr,nc=img.shape
    img=resize(img, (2000, 2000),anti_aliasing=True)
    img_resized = resize_image(img)
    lp, _ = laplacian_pyramid(img_resized, L)
    lp = enhance_coefficients(lp, L, params)
    rs = reconstruct_image(lp, L)
    rs = rs[:img.shape[0], :img.shape[1]]
    rs=resize(rs, (nr, nc),anti_aliasing=True)
    rs = (rs-np.min(rs.flatten()))/(np.max(rs.flatten())-np.min(rs.flatten()))
    rs=np.uint16(rs*255.0)
    return rs
