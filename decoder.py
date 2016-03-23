__author__ = 'anton'
import numpy as np
from affine_transformer import get_affine_transform
import time
import multiprocessing as mp
from functools import partial


def _downsample_locally(start_x, start_y, domain_size, range_size, image):
    block = np.zeros((range_size, range_size))
    # neighbours count
    nc = domain_size/range_size
    for y in xrange(range_size):
        domain_pos_y = start_y+y*nc
        for x in xrange(range_size):
            domain_pos_x = start_x+x*nc
            block[y, x] = int(image[domain_pos_y:domain_pos_y+nc, domain_pos_x:domain_pos_x+nc].sum()/(nc**2))
    return block


def decode_parallel(iterations, width, height, channels_transformations):
    pool = mp.Pool()
    start_time = time.time()
    # can not use this with pool in _decode_channel
    result = pool.map(
        partial(_decode_channel, iter=iterations, width=width, height=height), channels_transformations)
    print time.time() - start_time
    return result


def _decode_channel(channel_transformations, iter, width, height):
    image = np.zeros((width, height), dtype=np.uint8)
    image.fill(127)
    pool = mp.Pool()
    transformations = []
    for transformation in channel_transformations:
        if transformation.is_flat:
            ry, rx = transformation.range_y, transformation.range_x
            rs = transformation.range_size
            offset = transformation.offset
            image[ry:ry+rs, rx:rx+rs] = offset
        else:
            transformations.append(transformation)

    for i in xrange(iter):
        mappings = pool.map(partial(_get_contractive_mapping, image=image), transformations)
        for rx, ry, rs, mapping in mappings:
            image[ry: ry+rs, rx:rx+rs] = mapping
        # for transformation in channel_transformations:
            # ry, rx = transformation.range_y, transformation.range_x
            # dy, dx = transformation.domain_y, transformation.domain_x
            # rs, ds = transformation.range_size, transformation.domain_size
            # scale, offset, type = transformation.scale, transformation.offset, transformation.transform_type
            # # TODO: try to avoid downsampling every time
            # downsampled = _downsample_locally(dx, dy, ds, rs, image)
            # image[ry:ry+rs, rx:rx+rs] =\
            #     get_affine_transform(0, 0, scale, offset, type, rs, downsampled)
    return image


def _get_contractive_mapping(transformation, image):
    ry, rx = transformation.range_y, transformation.range_x
    dy, dx = transformation.domain_y, transformation.domain_x
    rs, ds = transformation.range_size, transformation.domain_size
    scale, offset, type = transformation.scale, transformation.offset, transformation.transform_type
    downsampled = _downsample_locally(dx, dy, ds, rs, image)
    return rx, ry, rs, get_affine_transform(0, 0, scale, offset, type, rs, downsampled)


def decode(iterations, width, height, channels_transformations):
    channels = []
    start_time = time.time()
    # run through all channels
    for channel_transformations in channels_transformations:
        # use mid-gray picture as initial
        # image = np.zeros((width, height), dtype=np.uint8)
        # image.fill(127)
        # for i in xrange(iterations):
        #     for transformation in channel_transformations:
        #         ry, rx = transformation.range_y, transformation.range_x
        #         dy, dx = transformation.domain_y, transformation.domain_x
        #         rs, ds = transformation.range_size, transformation.domain_size
        #         scale, offset, type = transformation.scale, transformation.offset, transformation.transform_type
        #         # TODO: try to avoid downsampling every time
        #         downsampled = _downsample_locally(dx, dy, ds, rs, image)
        #         image[ry:ry+rs, rx:rx+rs] =\
        #             get_affine_transform(0, 0, scale, offset, type, rs, downsampled)
        image = _decode_channel(channel_transformations, iterations, width, height)
        channels.append(image)
    print time.time() - start_time
    return channels
