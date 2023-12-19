# segmentation.py

import numpy as np

def segmentation(*argv):
    image, filter, smooth, threshold = argv

    if filter == 1:
        px_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        py_kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        gradient_x = np.convolve(image, px_kernel)
        gradient_y = np.convolve(image, py_kernel)

        result = gradient_x, gradient_y
    
    elif filter == 2:
        sobel_x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        gradient_x = np.convolve(image, sobel_x_kernel)
        gradient_y = np.convolve(image, sobel_y_kernel)

        result = gradient_x, gradient_y

    elif filter == 3:
        roberts_45_kernel = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]])
        roberts_minus_45_kernel = np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]])

        gradient_45 = np.convolve(image, roberts_45_kernel)
        gradient_minus_45 = np.convolve(image, roberts_minus_45_kernel)

        result = gradient_45, gradient_minus_45

    elif filter == 4:
        prewitt_45_kernel = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
        prewitt_minus_45_kernel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])

        gradient_45 = np.convolve(image, prewitt_45_kernel)
        gradient_minus_45 = np.convolve(image, prewitt_minus_45_kernel)

        result = gradient_45, gradient_minus_45

    elif filter == 5:
        sobel_45_kernel = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
        sobel_minus_45_kernel = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])

        gradient_45 = np.convolve(image, sobel_45_kernel)
        gradient_minus_45 = np.convolve(image, sobel_minus_45_kernel)

        result = gradient_45, gradient_minus_45

    elif filter == 6:
        compass_kernels = [
            np.array([[1, 1, 1], [1, -2, 1], [-1, -1, -1]]),
            np.array([[1, 1, 1], [-1, -2, 1], [-1, -1, 1]]),

            np.array([[1, 1, 1], [-1, -2, -1], [-1, 1, 1]]),
            np.array([[-1, 1, 1], [-1, -2, -1], [-1, 1, 1]]),

            np.array([[-1, -1, 1], [-1, -2, 1], [1, 1, 1]]),
            np.array([[-1, -1, -1], [1, -2, 1], [1, 1, 1]]),

            np.array([[1, -1, -1], [1, -2, -1], [1, 1, 1]]),
            np.array([[1, 1, -1], [1, -2, -1], [1, 1, -1]]),
        ]

        ans = np.zeros_like(image)

        for kernel in compass_kernels:
            ans = np.convolve(image, kernel)
            ans = np.maximum(ans, ans)

        magnitude = np.sqrt(np.sum(ans**2, axis=-1))

        threshold_image = (magnitude > threshold).astype(np.uint8) * 255

        result = threshold_image

    return result