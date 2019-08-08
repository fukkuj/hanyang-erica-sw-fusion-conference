import numpy as np


def _compute_image_x_gradients(slice):
    """
    compute x-axis gradient of image.

    Arguments:
    ----------
    :slice slice of images, shaped of (batch_size, channel, height, width)

    Returns:
    --------
    x-axis gradients which is result of x-convolution
    """

    # number of images, number of channels
    n, c = np.shape(slice)[:2]
    
    # kernel for computing gradient
    kernel_x = [
        [-2, -1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
        [-2, -1, 0, 1, 2],
    ]

    # compute gradient using convolution operation
    x_grad = np.sum(np.multiply(slice, np.array(kernel_x).reshape(1, 1, 5, 5)), axis=(2, 3))
    
    return x_grad.reshape(n, c, 1, 1)
    
def _compute_image_y_gradients(slice):
    """
    compute y-axis gradient of image

    Arguments:
    ----------
    :slice slice of images, shaped of (batch_size, channel, 5, 5)

    Returns:
    --------
    y-axis gradients which is result of y-convolution
    """
    
    # number of images, number of channels
    n, c = np.shape(slice)[:2]
    
    # kernel for computing y-axis gradients
    kernel_y = [
        [-2, -2, -2, -2, -2],
        [-1, -1, -1, -1, -1],
        [ 0,  0,  0,  0,  0],
        [ 1,  1,  1,  1,  1],
        [ 2,  2,  2,  2,  2]
    ]
    
    # compute gradient using convolution operation
    y_grad = np.sum(np.multiply(slice, np.array(kernel_y).reshape(1, 1, 5, 5)), axis=(2, 3))
    
    return y_grad.reshape(n, c, 1, 1)
    
def _compute_image_gradients(slice):
    """
    compute gradient with respect to x, y axis.

    Arguments:
    ----------
    :slice slice of images, shaped of (batch_size, channel, 5, 5)

    Returns:
    --------
    :res result of x and y convolution
    """

    n, c = np.shape(slice)[:2]
    res = np.zeros((n, c*2, 1, 1))
    
    # compute x and y convolution to compute gradients
    x_grad = _compute_image_x_gradients(slice)
    y_grad = _compute_image_y_gradients(slice)
    
    # stack x-axis convolution and y-axis convolution into single array
    res[:, :c] = x_grad
    res[:, c:] = y_grad
    
    return res.squeeze()

def compute_image_gradients(images):
    """
    compute gradients of images

    Arguments:
    ----------
    :images images, shaped of (batch_size, channel, height, width)

    Returns:
    --------
    :res result of convolution
    """

    n, c, h, w = np.shape(images)
    zero_padded = np.zeros((n, c, h+4, w+4))
    zero_padded[:, :, 2:h+2, 2:w+2] = images
    
    res = np.zeros((n, 2*c, h, w))
    
    # convolution
    for _h in range(h):
        for _w in range(w):
            slice = zero_padded[:, :, _h:_h+5, _w:_w+5].reshape(n, c, 5, 5)
            res[:, :, _h, _w] = _compute_image_gradients(slice)
            
    return res
