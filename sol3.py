import numpy as np
from scipy import ndimage
from imageio import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    :param im:  a grayscale image with double values in [0, 1]
    :param max_levels:  the maximal number of levels1 in the resulting pyramid.
    :param filter_size:  the size of the Gaussian filter
    :return:  a gaussian pyramid

    """
    binom_gaus = np.array([1, 1])
    filtered_vec, nor_factor = get_filtered_vec(filter_size, binom_gaus)
    if nor_factor != 1:
        filtered_vec = np.array([filtered_vec]) * (1 / nor_factor)
    else:
        filtered_vec = np.array([1])
    pyr = list()
    if max_levels == 1:
        return [im], filtered_vec
    pyr.append(im)
    i = 1
    new_img = np.copy(im)
    transpose_filter = filtered_vec.T
    while i < max_levels and (new_img.shape[0] / 2 >= 16) and new_img.shape[1] / 2 >= 16:
        reduce = ndimage.convolve(ndimage.convolve(new_img, filtered_vec), transpose_filter)
        reduce = reduce[::2, ::2]
        pyr.append(reduce)
        new_img = reduce.copy()
        i = i + 1
    return pyr, filtered_vec


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels:  the maximal number of levels1 in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
    :return: a laplacian pyramid
    """
    gaussian_lev, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    pyr = list()
    len1 = len(gaussian_lev) - 1

    for i in range(len1 + 1):
        if i == len1:
            pyr.append(gaussian_lev[len1])
        else:
            shape1 = gaussian_lev[i + 1].shape
            x, y = shape1
            pading_zero = np.zeros(((x * 2), (y * 2)))
            pading_zero[::2, ::2] = gaussian_lev[i + 1]
            expanded = ndimage.convolve(ndimage.convolve(pading_zero, filter_vec * 2), (filter_vec * 2).T)
            laplacan_pyr = gaussian_lev[i] - expanded
            pyr.append(laplacan_pyr)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """
    :param lpyr: Laplacian pyramid
    :param filter_vec: the filter
    :param coeff:f is a python list. The list length is the same as the number of levels in the pyramid lpyr
    :return: reconstruction of an image from its Laplacian Pyramid
    """
    for i in range(len(lpyr)):
        lpyr[i] *= coeff[i]
    j = len(lpyr) - 1
    to_expand = lpyr[j]
    while j != 0:
        x_len, y_len = to_expand.shape
        expand = np.zeros(((x_len * 2), (y_len * 2)))
        expand[::2, ::2] = to_expand
        final_expand = ndimage.convolve(ndimage.convolve(expand, filter_vec * 2), (filter_vec * 2).T)
        sum_image = final_expand + lpyr[j - 1]
        to_expand = sum_image
        j -= 1
    return to_expand


def get_filtered_vec(filtered_size, binom_gaus):
    """
    :param filtered_size: the size of the Gaussian filter
    :return: the resulting pyramid pyr and filter_vec which is row vector of shape (1, filter_size) used for the pyramid construction
    and the factor to normalization
    """
    norm = 2 ** (filtered_size - 1)
    if filtered_size - 1 <= 0:
        return np.array([1]), 1
    first_cov = np.array([1, 1])
    i = 2
    while i != filtered_size:
        first_cov = np.convolve(first_cov, binom_gaus)
        i += 1
    return first_cov, norm


def render_pyramid(pyr, levels):
    high = pyr[0].shape[0]
    colons = 0
    # lst_widths = [pyramid.shape[1] for pyramid in pyr]
    lst_widths = [pyr[i].shape[1] for i in range(levels)]
    updated = np.zeros((high, sum(lst_widths)))
    numlevel = min(levels, len(pyr))
    i = 0
    for p in pyr:
        if i == levels:
            return np.array(updated)
        im = (p - p.min()) / (p.max() - p.min())
        updated[0: p.shape[0], colons: p.shape[1] + colons] = im
        colons += im.shape[1]
        i += 1
    return np.array(updated)


def display_pyramid(pyr, levels):
    """
     display the stacked pyramid image
    :param pyr: a Gaussian or Laplacian pyramid
    :param levels:  number of levels
    """
    res = render_pyramid(pyr, levels)
    plt.imshow(res, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    """
    :param im1:  grayscale images to be blended.
    :param im2:  grayscale images to be blended.
    :param mask:  is a boolean (i.e. dtype == np.bool) mask containing True and False representing which parts of im1,2
    :param max_levels: is the max_levels parameter you should use when generating the Gaussian and Laplacian pyramids
    :param filter_size_im:  is the size of the Gaussian filter
    :param filter_size_mask: is the size of the Gaussian filter
    :return: the blending image
    """
    lap_im1, filter1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    lap_im2, filter2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gaussian_mask, filter3 = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    pyramid_blended = [cal_blend_k(gaussian_mask, i, lap_im1, lap_im2) for i in range(max_levels)]
    get_blending = laplacian_to_image(pyramid_blended, filter1, [1 for i in range(max_levels)])
    return np.clip(get_blending, 0, 1)


def cal_blend_k(gaussian_mask, i, lap_im1, lap_im2):
    """ calculate a specific lev in the blending laplacian"""
    mask_first_factor = gaussian_mask[i]
    mask_sec_factor = (1 - gaussian_mask[i])
    img1_mask = mask_first_factor * lap_im1[i]
    img2_mask = mask_sec_factor * lap_im2[i]
    laplacian_blend_i = img1_mask + img2_mask
    return laplacian_blend_i


def read_image(filename, representation):
    """
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: representation code, either 1 or 2 defining whether the output should be a grayscale
                           image (1) or an RGB image (2).
    :return:    function which reads an image file and converts it into a given representation  and returns it.
    """
    im = imread(filename)  # imread return a numpy array

    # convert rgb to grayscale
    if representation == 1 and len(im.shape) == 3:
        return convert_rgb_to_grayscale(im)
    # normalized grayscale
    if representation == 1:
        im = im / 255
        return im.astype(np.float64)
    else:
        # convert to rgb
        return (im / 255).astype(np.float64)


def convert_rgb_to_grayscale(image):
    """
    :param image: image
    :return: a grayscale image
    """
    im = rgb2gray(image).astype(np.float64)
    return im


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


def blending_example1():
    image2 = read_image(relpath("lion.jpg"), 2)
    image1 = read_image(relpath('cat.jpg'), 2)
    mask_image = read_image(relpath("catmask.jpg"), 1)
    copy_im = image1.copy()
    r1, r2, g1, g2= image2[:, :, 0], image1[:, :, 0], image2[:, :, 1], image1[:, :, 1]
    b1 ,b2= image2[:, :, 2],image1[:, :, 2]
    copy_im[:, :, 0] = pyramid_blending(r1, r2, mask_image, 5, 3, 3)
    copy_im[:, :, 1] = pyramid_blending(g1, g2, mask_image, 5, 3, 3)
    copy_im[:, :, 2] = pyramid_blending(b1, b2, mask_image, 5, 3, 3)
    subp = plt.subplots(nrows=2, ncols=2)
    subp[1][0, 0].imshow(image1)
    subp[1][0, 1].imshow(image2)
    subp[1][1, 0].imshow(mask_image)
    subp[1][1, 1].imshow(copy_im)
    plt.show()
    mask_image = mask_image.astype(np.bool)
    return image1, image2, mask_image, copy_im


def blending_example2():
    image1 = read_image(relpath("lion.jpg"), 2)
    image2 = read_image(relpath("makha.jpg"), 2)
    mask_image = read_image(relpath("maskmask.jpg"), 1)
    copy_im = image1.copy()
    r1, r2, g1, g2 = image2[:, :, 0], image1[:, :, 0], image2[:, :, 1], image1[:, :, 1]
    copy_im[:, :, 0] = pyramid_blending(r1, r2, mask_image, 5, 3, 3)
    copy_im[:, :, 1] = pyramid_blending(g1,g2, mask_image, 5, 3, 3)
    b1, b2 = image2[:, :, 2], image1[:, :, 2]
    copy_im[:, :, 2] = pyramid_blending(b1, b2, mask_image, 5, 3, 3)
    subp = plt.subplots(nrows=2, ncols=2)
    subp[1][0, 0].imshow(image1)
    subp[1][0, 1].imshow(image2)
    subp[1][1, 0].imshow(mask_image)
    subp[1][1, 1].imshow(copy_im)
    plt.show()
    mask_image = mask_image.astype(np.bool)
    return image1, image2, mask_image, copy_im


