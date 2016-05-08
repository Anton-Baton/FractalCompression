import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
import pyopencl.array as cl_array
from transformations import Transformation
import time


RANGE_BLOCK_SIZE = 8


def get_downsampled(block, width, height):
    """
    Downsample channel averaging neighbour values.
    Expect that width and height are integer multipliers of 2
    :param block: image data
    :param width: image width
    :param height: image height
    :return: downsampled blocks values
    """

    # number of neighbour cell(neighbours number) to be averaged
    nc = 2
    downsampled_blocks = np.zeros((width/nc, width/nc))
    for y in xrange(0, height, nc):
        for x in xrange(0, width, nc):
            # compute real coordinates
            value = int(block[y: y+nc, x: x+nc].sum()/(nc**2))
            downsampled_blocks[y/nc, x/nc] = value
    return downsampled_blocks


def get_global_size(image_width, image_height, range_block_size):
    return image_width/range_block_size, image_height/range_block_size


def get_range_blocks_coordinates(image_width, image_height, range_block_size):
    range_blocks = []
    for i in xrange(0, image_height-range_block_size+1, range_block_size):
        for j in xrange(0, image_width-range_block_size+1, range_block_size):
            range_blocks.append(cl_array.vec.make_int2(j, i))
    return np.array(range_blocks)


def encode(img):
    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    kernel = open('image_processing_kernel.cl', 'rb').read()

    program = cl.Program(ctx, kernel).build(options=['-D', 'range_block_size={0}'.format(RANGE_BLOCK_SIZE)])
    cl_image_format = cl.ImageFormat(cl.channel_order.R,
                                   cl.channel_type.UNSIGNED_INT8)
    print cl_image_format.channel_count
    global_size = get_global_size(*img.size, range_block_size=RANGE_BLOCK_SIZE)

    transfomation_struct = np.dtype([('domain_x', np.int32), ('domain_y', np.int32),
                                     ('range_x', np.int32), ('range_y', np.int32),
                                     ('scale', np.float32), ('offset', np.int32),
                                     ('symmetry', np.int32)])
    transfomation_struct, cdecl = cl_tools.match_dtype_to_c_struct(device, 'transformation', transfomation_struct)
    cl_tools.get_or_register_dtype('transformation', transfomation_struct)

    channels_transformations = []
    downsampled_size = (img.size[0]/2, img.size[1]/2)
    start_time = time.time()
    range_block_coordinates = get_range_blocks_coordinates(img.size[1], img.size[0], RANGE_BLOCK_SIZE)
    for i, channel in enumerate(img.image_data):
        buffer = channel.astype(np.uint8)
        downsampled = get_downsampled(channel, img.size[0], img.size[1]).astype(np.uint8)

        init_image = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              cl_image_format, img.size, None, buffer)
        downsampled_image = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     cl_image_format, downsampled_size, None, downsampled)

        range_coordinates = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                      hostbuf=range_block_coordinates)

        transformations = np.zeros((global_size[0]*global_size[1],), transfomation_struct)
        out_transformations_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, transformations.nbytes)

        program.find_matches.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32, np.int32, None])
        find_matches_event = program.find_matches(queue, (global_size[0]*global_size[1], 1), None, init_image,
                                                  downsampled_image, range_coordinates, np.int32(img.size[1]),
                                                  np.int32(img.size[0]), np.int32(global_size[1]),
                                                  out_transformations_buffer)

        got_transformations_event = cl.enqueue_copy(queue, transformations, out_transformations_buffer,
                                                    wait_for=[find_matches_event])
        got_transformations_event.wait()

        transforamtions_list = []
        for tr in transformations:
            transforamtions_list.append(Transformation.from_struct(RANGE_BLOCK_SIZE, RANGE_BLOCK_SIZE*2, tr))
        channels_transformations.append(transforamtions_list)
    print time.time() - start_time
    return channels_transformations
