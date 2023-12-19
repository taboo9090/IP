# generics.py

import numpy as np

def arithmetic_mean_filter(kernel_size, image):
    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = np.sum(kernel)/(kernel_size**2)

    return result_image

def geometric_mean_filter(kernel_size, image):
    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = (np.prod(kernel)**(1/(kernel_size**2)))

    return result_image

def harmonic_mean_filter(kernel_size, image):
    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = (kernel_size**2) / np.sum(1/kernel)

    return result_image

def contra_harmonic_filter(kernel_size, image, q_value):
    zero_pixel_num = image[image == 0].shape[0]
    percent = (zero_pixel_num / (image.shape[0] * image.shape[1])) * 100

    if q_value < 0 and percent > 3:
        raise ValueError("Contra Harmonic function doesn't support pepper noise")

    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = (np.sum(kernel)**(q_value + 1)) / (np.sum(kernel)**q_value)
            
    return result_image

def alpha_trimmed_mean_filter(kernel_size, image, d_value):
    if d_value >= (kernel_size**2) - 1:
        raise ValueError("d is greater than (m * n) - 1")
    
    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = (1 / (kernel_size**2)) * np.sum(kernel)
            
    return result_image

def min_filter(kernel_size, image):
    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = np.min(kernel)
            
    return result_image

def max_filter(kernel_size, image):
    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = np.max(kernel)
            
    return result_image

def median_filter(kernel_size, image):
    image = np.pad(array=image, pad_width=int(kernel_size/2), mode="constant", constant_values=0)

    R, C = image.shape
    result_image = np.zeros([R, C])

    for i in range(0, R):
        for j in range(0, C):
            kernel = image[i:i+kernel_size, j:j+kernel_size]
            result_image[i, j] = 1/2 * (np.max(kernel) + np.min(kernel))
            
    return result_image


def proccess(flag, *argv):
    kernel_size, image, param = argv

    filters = {
        "1": arithmetic_mean_filter(kernel_size, image),
        "2": geometric_mean_filter(kernel_size, image),
        "3": harmonic_mean_filter(kernel_size, image),
        "4": contra_harmonic_filter(kernel_size, image, param),
        "5": alpha_trimmed_mean_filter(kernel_size, image, param),
        "6": min_filter(kernel_size, image),
        "7": max_filter(kernel_size, image),
        "8": median_filter(kernel_size, image)
    }

    return filters[flag]