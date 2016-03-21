__author__ = 'anton'
import numpy as np

# types of transformations
TRANSFORM_NONE = 0
TRANSFORM_ROTATE_90 = 1
TRANSFORM_ROTATE_180 = 2
TRANSFORM_ROTATE_270 = 3
TRANSFORM_FLIP_H = 4
TRANSFORM_FLIP_V = 5
TRANSFORM_FLIP_FORWARD_DIAGONAL = 6
TRANSFORM_FLIP_REVERSE_DIAGONAL = 7
TRANSFORM_MAX = 8


def _is_in_scanline_order(type):
    return type in [TRANSFORM_NONE, TRANSFORM_ROTATE_180, TRANSFORM_FLIP_H, TRANSFORM_FLIP_V]


def _is_positive_x(type):
    return type in [TRANSFORM_NONE, TRANSFORM_ROTATE_90, TRANSFORM_FLIP_V, TRANSFORM_FLIP_REVERSE_DIAGONAL]


def _is_positive_y(type):
    return type in [TRANSFORM_NONE, TRANSFORM_ROTATE_270, TRANSFORM_FLIP_H, TRANSFORM_FLIP_REVERSE_DIAGONAL]


def get_affine_transform(start_x, start_y, scale, offset, type, range_block_size, downsampled, normalize=False):
    """
    Produce affine transformation of specified type over the specified downsampled block
    :param start_x: domain top left x
    :param start_y: domain top left y
    :param scale: affine transformation scale
    :param offset: affine transformation offset
    :param type: affine transformation type (one of 8 specified)
    :param range_block_size: size of target range block
    :param downsampled: downsampled image data
    :param normalize: specify if domain block should be normalized by it`s average
    :return: transformed domain block
    """

    # start_x and start_y - coordinates in original picture. In downsampled it is twice less
    # start_points
    start_x /= 2
    start_y /= 2

    # move by x and y
    dx = dy = 1

    # order in which symbols appear on the screen (left to right, top to bottom)
    # controls the moving of current element pointer
    scanline_order = _is_in_scanline_order(type)
    # specify if pointer is moved left to right or right to left
    if not _is_positive_x(type):
        dx = -1
        start_x += range_block_size-1
    # specify if pointer is moved top to bottom or bottom to top
    if not _is_positive_y(type):
        dy = -1
        start_y += range_block_size-1

    x = y = 0
    block = np.zeros((range_block_size, range_block_size))
    block_sum = 0
    # run through all 4 pixel downsampled mini-blocks and store them in special order
    for i in xrange(range_block_size):
        for j in xrange(range_block_size):
            pixel = max(0, min(255, int(scale*downsampled[start_y+y, start_x+x])+offset))
            block[i, j] = pixel
            #block_sum += pixel
            if scanline_order:
                x += dx
            else:
                y += dy
        if scanline_order:
            x = 0
            y += dy
        else:
            y = 0
            x += dx
    #average_value = block_sum/(range_block_size**2)
    return block #- average_value, average_value if normalize else block.ravel(), average_value
