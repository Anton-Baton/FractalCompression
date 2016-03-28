__author__ = 'anton'
import numpy as np
from transformations import Transformation
from affine_transformer import get_affine_transform, TRANSFORM_NONE, TRANSFORM_MAX
import time


RANGE_SIZES = [32, 16, 8, 4]
RANGE_BLOCK_SIZE = RANGE_SIZES[0]
DOMAIN_SCALE_FACTOR = 2
DOMAIN_SIZES = [DOMAIN_SCALE_FACTOR*x for x in RANGE_SIZES]
DOMAIN_BLOCK_SIZE = DOMAIN_SIZES[0]
DOMAIN_SKIP_FACTOR = 8


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


def _find_domain_block(range_x, range_y, range_size, variance_treshold, channel, domain_pool, domain_averages):
    """
    Function finds appropriate domain block for a specified range block
    :param range_x: range block top left x coordinate
    :param range_y: range block top left y coordinate
    :param variance_treshold: boundary of the flat block variance
    :param channel: image data
    :param domain_pool: domain pool
    :return: tuple of domain block index, scale and offset(domain block index == -1 signals that block is flat)
    """
    # extract range block from image data
    range_block = channel[range_y:range_y+range_size, range_x:range_x+range_size].copy()
    # normalize range block by average pixel value
    range_block_average = int(range_block.sum()/(range_size**2))

    # prepare values to store result
    min_error = 1e9
    min_block_index = -1
    min_scale_factor = 0
    min_offset = range_block_average

    range_block_variance = np.sum(range_block**2-range_block_average**2)/(range_size**2)
    if range_block_variance > variance_treshold:
        # search of appropriate domain block
        range_block -= range_block_average
        bottom = 0
        domain_block_average = 0
        for domain_block_index, domain_block in enumerate(domain_pool):
            # find scale, offset and MSE
            if domain_block_index % TRANSFORM_MAX == 0:
                domain_block_average = domain_averages[domain_block_index]
                avg_domain_block = domain_block - domain_block_average
                bottom = (avg_domain_block**2).sum()
            if bottom == 0.0:
                scale = 0.0
            else:
                scale = (range_block * avg_domain_block).sum()*1.0/bottom

            offset = int(range_block_average-domain_block_average*scale)
            difference = (range_block - scale*avg_domain_block)
            error = (difference * difference).sum()*1.0/(range_size**2)
            if error < min_error:
                min_error = error
                min_block_index = domain_block_index
                min_scale_factor = scale
                min_offset = offset
    return min_block_index, min_scale_factor, min_offset, min_error


def _get_domain_pool(width, height, divergence_treshold, downsampled,
                     domain_block_size, domain_skip_factor, range_block_size):
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
    for y in xrange(0, height-domain_block_size+1, domain_skip_factor):
        for x in xrange(0, width-domain_block_size+1, domain_skip_factor):
            for transform_type in xrange(TRANSFORM_NONE, TRANSFORM_MAX):
                domain = get_affine_transform(
                    x, y, 1.0, 0, transform_type, range_block_size, downsampled)
                domain_average = int(domain.sum()/(range_block_size**2))
                # unfortunately, analyser convinced that this is int not an array
                domain_divergence = np.sum(domain**2 - domain_average**2)/(range_block_size**2)
                if domain_divergence > divergence_treshold:
                    domain_pool.append(domain)
                    domain_averages.append(domain_average)
                    domain_positions.append((x, y))
    return domain_pool, domain_averages, domain_positions


def _get_filtered_domain_pool(width, height, domain_sizes, range_sizes, downsampled, domain_skip_factor):
    domain_pool, domain_avg, domain_pos = _get_quadtree_domain_pool(
        width, height, domain_sizes, range_sizes, downsampled, domain_skip_factor)
    pass


def _get_quadtree_domain_pool(width, height, variance_treshold, domain_sizes, range_sizes, downsampled, domain_skip_factor):
    domain_pool = {}
    domain_averages = {}
    domain_positions = {}
    for rbs, dbs in zip(range_sizes, domain_sizes):
        dpool = []
        davg = []
        dpos = []
        variances = []

        num = 0
        # TODO : check the domain skip factor
        for y in xrange(0, height-dbs+1, max(dbs/2, domain_skip_factor)):
            for x in xrange(0, width-dbs+1, max(dbs/2, domain_skip_factor)):
                domain_average = 0
                for transform_type in xrange(TRANSFORM_NONE, TRANSFORM_MAX):
                    domain = get_affine_transform(
                        x, y, 1.0, 0, transform_type, rbs, downsampled)
                    if transform_type == TRANSFORM_NONE:
                        domain_average = int(domain.sum()/(rbs**2))
                        domain_variance = np.sum(domain**2-domain_average**2)/(rbs**2)
                        variances.append(domain_variance)
                        if domain_variance < variance_treshold*rbs:
                            num += 1
                            break
                    dpool.append(domain)
                    davg.append(domain_average)
                    dpos.append((x, y))
        domain_pool[dbs] = dpool
        domain_averages[dbs] = davg
        domain_positions[dbs] = dpos
        print num, len(dpool)
        print np.min(variances), np.max(variances), np.mean(variances)
    return domain_pool, domain_averages, domain_positions


def _get_range_block_transformations(rx, ry, range_size, domain_size, error_treshold, variance_treshold,
                                     channel, domain_pool, domain_positions, domain_averages):
    is_flat = False
    transformations = []
    domain_index, scale, offset, error = \
        _find_domain_block(rx, ry, range_size, variance_treshold, channel,
                           domain_pool[domain_size], domain_averages[domain_size])
    if domain_index == -1:
        is_flat = True
        domain_index = 0
    # TODO: remove dependency from RANGE_SIZES
    if not is_flat and error > error_treshold and range_size/2 in RANGE_SIZES:
        # TODO: eliminate code repeat
        # new range block size
        nrbs = range_size/2
        # new domain block size
        ndbs = domain_size/2
        lt = _get_range_block_transformations(rx, ry, nrbs, ndbs, error_treshold, variance_treshold,
                                              channel, domain_pool, domain_positions, domain_averages)
        rt = _get_range_block_transformations(rx+nrbs, ry, nrbs, ndbs, error_treshold, variance_treshold,
                                              channel, domain_pool, domain_positions, domain_averages)
        lb = _get_range_block_transformations(rx, ry+nrbs, nrbs, ndbs, error_treshold, variance_treshold,
                                              channel, domain_pool, domain_positions, domain_averages)
        rb = _get_range_block_transformations(rx+nrbs, ry+nrbs, nrbs, ndbs, error_treshold, variance_treshold,
                                              channel, domain_pool, domain_positions, domain_averages)
        transformations.extend(lt)
        transformations.extend(rt)
        transformations.extend(lb)
        transformations.extend(rb)
    else:
        domain_x, domain_y = domain_positions[domain_size][domain_index]
        transformation_type = domain_index % TRANSFORM_MAX
        transformation = Transformation(domain_x, domain_y, rx, ry, scale, offset, range_size,
                                        domain_size, transformation_type, is_flat)
        transformations = [transformation]
    return transformations


def encode(img):
    """
    Function that find the appropriate transformations for all range blocks
    :param img: ImageData object that contain info about image
    :return: transformations
    """
    channels_transformations = []
    start_time = time.time()
    range_variance_treshold = 10
    domain_variance_treshold = 10
    error_treshold = 100
    for channel in img.image_data:
        # downsample data to compare domain and range blocks
        downsampled = _downsample_by_2(channel, img.width, img.height)
        start_time1 = time.time()
        domain_pool, domain_avg, domain_positions = \
            _get_quadtree_domain_pool(img.width, img.height, domain_variance_treshold,
                                      DOMAIN_SIZES, RANGE_SIZES, downsampled, DOMAIN_SKIP_FACTOR)
        print time.time() - start_time1
        transformations = []
        #domain_pool = domain_pool[:256]
        # run through all range blocks and find appropriate domain block
        # this implementation find domain block with the least error
        # TODO: make iteration more clear
        print 'prepared'
        for y in (range(0, img.height-RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE-1)+[img.height-RANGE_BLOCK_SIZE]):
            for x in (range(0, img.width-RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE-1)+[img.width-RANGE_BLOCK_SIZE]):

                transformations.extend(
                    _get_range_block_transformations(x, y, RANGE_BLOCK_SIZE, DOMAIN_BLOCK_SIZE, error_treshold,
                                                     range_variance_treshold, channel, domain_pool, domain_positions,
                                                     domain_avg))
                print '.',
            print '\n'
        channels_transformations.append(transformations)
    print time.time() - start_time
    return channels_transformations



