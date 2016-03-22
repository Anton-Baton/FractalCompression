__author__ = 'anton'
import numpy as np
from affine_transformer import get_affine_transform
import time


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


def decode(iterations, width, height, channels_transformations):
    channels = []
    start_time = time.time()
    # run through all channels
    for channel_transformations in channels_transformations:
        # use mid-gray picture as initial
        image = np.zeros((width, height), dtype=np.uint8)
        image.fill(127)
        for i in xrange(iterations):
            for transformation in channel_transformations:
                ry, rx = transformation.range_y, transformation.range_x
                dy, dx = transformation.domain_y, transformation.domain_x
                rs, ds = transformation.range_size, transformation.domain_size
                scale, offset, type = transformation.scale, transformation.offset, transformation.transform_type
                # TODO: try to avoid downsampling every time
                downsampled = _downsample_locally(dx, dy, ds, rs, image)
                image[ry:ry+rs, rx:rx+rs] =\
                    get_affine_transform(0, 0, scale, offset, type, rs, downsampled)
        channels.append(image)
    print time.time() - start_time
    return channels
