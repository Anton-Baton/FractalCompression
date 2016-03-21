__author__ = 'anton'
import numpy as np
from affine_transformer import get_affine_transform
import time


# def transform_domain_block(domain_x, domain_y, domain_size, range_size, scale, offset, image):
#     transformed_domain_block = np.zeros((range_size, range_size))
#     scale = domain_size/range_size
#     for i in xrange(range_size):
#         for j in xrange(range_size):
#             pixel = image[domain_y:domain_y+scale, domain_x:domain_x+scale].sum()/(scale**2)
#             transformed_domain_block[i, j] = scale*pixel+offset
#     return transformed_domain_block
def _downsample_locally(start_x, start_y, domain_size, range_size, image):
    block = np.zeros((range_size, range_size))
    # neighbours count
    nc = domain_size/range_size
    for y in xrange(range_size):
        for x in xrange(range_size):
            block[y, x] = image[start_y:start_y+nc, start_x: start_x+nc].sum()/(nc**2)
    return block


def decode(iterations, width, height, channels_transformations):
    channels = []
    start_time = time.time()
    # run through all channels
    num = 0
    for channel_transformations in channels_transformations:
        # use mid-gray picture as initial
        image = np.zeros((width, height))
        image.fill(127)
        for i in xrange(iterations):
            for transformation in channel_transformations:
                ry, rx = transformation.range_y, transformation.range_y
                dy, dx = transformation.domain_y, transformation.domain_x
                rs, ds = transformation.range_size, transformation.domain_size
                scale, offset, type = transformation.scale, transformation.offset, transformation.transform_type
                # TODO: try to avoid downsampling every time
                if image is None:
                    print num
                downsampled = _downsample_locally(dx, dy, ds, rs, image)
                image[ry:ry+rs, rx:rx+rs] =\
                    get_affine_transform(0, 0, scale, offset, type, rs, downsampled).reshape((rs, rs))
                num += 1
        channels.append(image)
    print time.time() - start_time
    return channels
