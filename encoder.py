__author__ = 'anton'
import numpy as np
from transformations import Transformation
from affine_transformer import get_affine_transform, TRANSFORM_NONE, TRANSFORM_MAX
import time


RANGE_BLOCK_SIZE = 8
DOMAIN_SCALE_FACTOR = 2
DOMAIN_BLOCK_SIZE = RANGE_BLOCK_SIZE*DOMAIN_SCALE_FACTOR
DOMAIN_SKIP_FACTOR = 4


def _downsample_by_2(channel, width, height):
    """
    Downsample channel averaging neighbour values.
    Expect that width and height are integer multipliers of 2
    :param channel: image data
    :param width: image width
    :param height: image height
    :return: downsampled blocks values
    """
    channel_len = len(channel)
    if channel_len != width*height:
        raise ValueError('channel length should be exactly width*height')

    # number of neighbour cell(neighbours number) to be averaged
    nc = 2
    downsampled_blocks = np.zeros((height/nc, width/nc))
    for y in xrange(0, height, nc):
        for x in xrange(0, width, nc):
            # compute real coordinates
            rc = y*width+x
            next_row_rc = rc+width
            value = sum(channel[rc: rc+nc]+channel[next_row_rc:next_row_rc+nc])/(nc**2)
            downsampled_blocks[y/nc, x/nc] = value
    return downsampled_blocks


def _find_domain_block(range_x, range_y, width, height, treshold, channel, domain_pool):
    """
    Function finds appropriate domain block for a specified range block
    :param range_x: range block top left x coordinate
    :param range_y: range block top left y coordinate
    :param width: image width
    :param height: image height
    :param treshold: search accuracy
    :param channel: image data
    :param domain_pool: domain pool
    :return: typle of domain block index, scale and offset
    """
    # extract range block from image data
    range_block = np.zeros((RANGE_BLOCK_SIZE**2,))
    for i in xrange(RANGE_BLOCK_SIZE):
        start_point = (range_y+i)*width+range_x
        # TODO: check the correctness of the next statement
        range_block[i*RANGE_BLOCK_SIZE:(i+1)*RANGE_BLOCK_SIZE] = channel[start_point: start_point+RANGE_BLOCK_SIZE]
    # normalize range block by average pixel value
    range_block_average = range_block.sum()/(RANGE_BLOCK_SIZE**2)
    range_block -= range_block_average

    # prepare values to store result
    min_error = 1e9
    min_block_index = -1
    min_scale_factor = 0
    min_offset = 0

    # search of appropriate domain block
    for domain_block_index, domain_block in enumerate(domain_pool):
        # find scale, offset and MSE
        scale = (range_block*domain_block).sum()/(domain_block**2)
        print scale
        difference = (range_block - scale*domain_block)
        offset = difference.sum()
        # TODO: prove this is working
        # error = ((scale*domain_block-range_block)**2).sum()/(RANGE_BLOCK_SIZE**2)
        error = (difference**2).sum()/(RANGE_BLOCK_SIZE**2)
        if error < min_error:
            min_error = error
            min_block_index = domain_block_index
            min_scale_factor = scale
            min_offset = offset

    return min_block_index, min_scale_factor, min_offset


def _get_domain_pool(width, height, downsampled):
    """
    Function that produce a domain pool with all domain blocks and their transformations
    :param width: width of initial image
    :param height: height of initial image
    :param downsampled: downsampled image date
    :return: domain pool and domain coordinated
    """
    domain_pool = []
    domain_positions = []
    # run through all domain blocks and create  transformations of
    for y in xrange(0, height-DOMAIN_BLOCK_SIZE+1, DOMAIN_SKIP_FACTOR):
        for x in xrange(0, width-DOMAIN_BLOCK_SIZE+1, DOMAIN_SKIP_FACTOR):
            for transform_type in xrange(TRANSFORM_NONE, TRANSFORM_MAX):
                domain_pool.append(get_affine_transform(
                    x, y, 1, 0, transform_type, RANGE_BLOCK_SIZE, downsampled, normalize=True))
                domain_positions.append((x, y))
    return domain_pool, domain_positions


def encode(img):
    """
    Function that find the appropriate transformations for all range blocks
    :param img: ImageData object that contain info about image
    :return: transformations
    """
    channels_transformations = []
    start_time = time.time()
    for channel in img.image_data:
        # downsample data to compare domain and range blocks
        downsampled = _downsample_by_2(channel, img.width, img.height)
        domain_pool, domain_positions = _get_domain_pool(img.width, img.height, downsampled)
        transformations = []
        # run through all range blocks and find appropriate domain block
        # this implementation find domain block with the least error
        for y in xrange(0, img.height, RANGE_BLOCK_SIZE):
            for x in xrange(0, img.width, RANGE_BLOCK_SIZE):
                domain_index, scale, offset = _find_domain_block(x, y, img.width, img.height, 500, channel, domain_pool)
                domain_x, domain_y = domain_positions[domain_index]
                transformation_type = domain_index % TRANSFORM_MAX
                transformations.append(Transformation(domain_x, domain_y, x, y, scale, offset,
                                                      RANGE_BLOCK_SIZE, DOMAIN_BLOCK_SIZE, transformation_type))
                print '.',
            print '\n'
        channels_transformations.append(transformations)
    print time.time() - start_time
    return channels_transformations



