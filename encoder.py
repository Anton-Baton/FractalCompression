__author__ = 'anton'
import numpy as np
from transformations import Transformation
from affine_transformer import get_affine_transform, TRANSFORM_NONE, TRANSFORM_MAX
import time


RANGE_BLOCK_SIZE = 8
DOMAIN_SCALE_FACTOR = 2
DOMAIN_BLOCK_SIZE = RANGE_BLOCK_SIZE*DOMAIN_SCALE_FACTOR
DOMAIN_SKIP_FACTOR = 16


def _downsample_by_2(channel, width, height):
    """
    Downsample channel averaging neighbour values.
    Expect that width and height are integer multipliers of 2
    :param channel: image data
    :param width: image width
    :param height: image height
    :return: downsampled blocks values
    """

    # number of neighbour cell(neighbours number) to be averaged
    nc = 2
    downsampled_blocks = np.zeros((height/nc, width/nc))
    for y in xrange(0, height, nc):
        for x in xrange(0, width, nc):
            # compute real coordinates
            value = int(channel[y: y+nc, x: x+nc].sum()/(nc**2))
            downsampled_blocks[y/nc, x/nc] = value
    return downsampled_blocks


def _find_domain_block(range_x, range_y, error_treshold, variance_treshold, channel, domain_pool, domain_averages):
    """
    Function finds appropriate domain block for a specified range block
    :param range_x: range block top left x coordinate
    :param range_y: range block top left y coordinate
    :param error_treshold: search accuracy. Unused by now
    :param variance_treshold: boundary of the flat block variance
    :param channel: image data
    :param domain_pool: domain pool
    :return: tuple of domain block index, scale and offset(domain block index == -1 signals that block is flat)
    """
    # extract range block from image data
    range_block = channel[range_y:range_y+RANGE_BLOCK_SIZE, range_x:range_x+RANGE_BLOCK_SIZE].copy()
    # normalize range block by average pixel value
    range_block_average = int(range_block.sum()/(RANGE_BLOCK_SIZE**2))
    range_block -= range_block_average

    # prepare values to store result
    min_error = 1e9
    min_block_index = -1
    min_scale_factor = 0
    min_offset = range_block_average

    range_block_variance = (range_block*range_block).sum()/(RANGE_BLOCK_SIZE**2)
    if range_block_variance > variance_treshold:
        # search of appropriate domain block
        for domain_block_index, domain_block in enumerate(domain_pool):
            # find scale, offset and MSE
            domain_block_average = domain_averages[domain_block_index]
            avg_domain_block = domain_block - domain_block_average
            bottom = (avg_domain_block * avg_domain_block).sum()
            if bottom == 0.0:
                scale = 0.0
            else:
                scale = (range_block * avg_domain_block).sum()*1.0/bottom

            offset = int(range_block_average-domain_block_average*scale)
            difference = (range_block - scale*avg_domain_block)
            error = (difference * difference).sum()*1.0/(RANGE_BLOCK_SIZE**2)
            if error < min_error:
                min_error = error
                min_block_index = domain_block_index
                min_scale_factor = scale
                min_offset = offset
    return min_block_index, min_scale_factor, min_offset


def _get_domain_pool(width, height, divergence_treshold, downsampled):
    """
    Function that produce a domain pool with all domain blocks and their transformations
    :param width: width of initial image
    :param height: height of initial image
    :param downsampled: downsampled image data
    :return: domain pool and domain coordinated
    """
    domain_pool = []
    domain_positions = []
    domain_averages = []
    # run through all domain blocks and create  transformations of
    for y in xrange(0, height-DOMAIN_BLOCK_SIZE+1, DOMAIN_SKIP_FACTOR):
        for x in xrange(0, width-DOMAIN_BLOCK_SIZE+1, DOMAIN_SKIP_FACTOR):
            for transform_type in xrange(TRANSFORM_NONE, TRANSFORM_MAX):
                domain = get_affine_transform(
                    x, y, 1.0, 0, transform_type, RANGE_BLOCK_SIZE, downsampled)
                domain_average = int(domain.sum()/(RANGE_BLOCK_SIZE**2))
                # unfortunately, analyser convinced that this is int not an array
                domain_divergence = np.sum((domain - domain_average)**2)/(RANGE_BLOCK_SIZE**2)
                if domain_divergence > divergence_treshold:
                    domain_pool.append(domain)
                    domain_averages.append(domain_average)
                    domain_positions.append((x, y))
    return domain_pool, domain_averages, domain_positions


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
        domain_pool, domain_avg, domain_positions = _get_domain_pool(img.width, img.height, 20, downsampled)
        transformations = []
        # domain_pool = domain_pool[:256]
        # run through all range blocks and find appropriate domain block
        # this implementation find domain block with the least error
        # TODO: make iteration more clear
        for y in (range(0, img.height-RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE-1)+[img.height-RANGE_BLOCK_SIZE]):
            for x in (range(0, img.width-RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE-1)+[img.width-RANGE_BLOCK_SIZE]):
                is_flat = False
                domain_index, scale, offset = \
                    _find_domain_block(x, y, 500, 20, channel, domain_pool, domain_avg)
                if domain_index == -1:
                    is_flat = True
                    domain_index = 0
                domain_x, domain_y = domain_positions[domain_index]
                transformation_type = domain_index % TRANSFORM_MAX
                transformations.append(Transformation(domain_x, domain_y, x, y, scale, offset, RANGE_BLOCK_SIZE,
                                                      DOMAIN_BLOCK_SIZE, transformation_type, is_flat))
                print '.',
            print '\n'
        channels_transformations.append(transformations)
    print time.time() - start_time
    return channels_transformations



