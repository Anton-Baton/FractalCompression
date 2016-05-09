import numpy as np
import pyopencl as cl
import pyopencl.tools as cl_tools
import pyopencl.array as cl_array
from transformations import Transformation
import time


RANGE_BLOCKS_SIZES = [16, 8, 4]
INITIAL_RANGE_BLOCK_SIZE = RANGE_BLOCKS_SIZES[0]

TRESHOLD = 50


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


def expand_coordinates(x, y, old_width):
    new_width = old_width/2
    delta = [(0, 0), (new_width, 0), (0, new_width), (new_width, new_width)]
    return [cl_array.vec.make_int2(x+dx, y+dy) for dx, dy in delta]


def encode(img):
    platform = cl.get_platforms()[0]
    device = platform.get_devices(cl.device_type.GPU)[0]
    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    kernel = open('image_processing_kernel.cl', 'rb').read()
    cl_image_format = cl.ImageFormat(cl.channel_order.R,
                                   cl.channel_type.UNSIGNED_INT8)

    transfomation_struct = np.dtype([('domain_x', np.int32), ('domain_y', np.int32),
                                     ('range_x', np.int32), ('range_y', np.int32),
                                     ('scale', np.float32), ('offset', np.int32),
                                     ('symmetry', np.int32), ('error', np.float32)])
    transfomation_struct, cdecl = cl_tools.match_dtype_to_c_struct(device, 'transformation', transfomation_struct)
    cl_tools.get_or_register_dtype('transformation', transfomation_struct)

    program = cl.Program(ctx, kernel).build(options=['-D', 'range_block_size={0}'.format(INITIAL_RANGE_BLOCK_SIZE)])

    programs_dict = {INITIAL_RANGE_BLOCK_SIZE: program}

    # global_size = get_global_size(*img.size, range_block_size=RANGE_BLOCK_SIZE)
    downsampled_size = (img.size[0]/2, img.size[1]/2)

    channels_transformations = []

    start_time = time.time()

    for i, channel in enumerate(img.image_data):
        buffer = channel.astype(np.uint8)
        downsampled = get_downsampled(channel, img.size[0], img.size[1]).astype(np.uint8)

        init_image = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                              cl_image_format, img.size, None, buffer)
        downsampled_image = cl.Image(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   cl_image_format, downsampled_size, None, downsampled)
        range_block_coordinates = get_range_blocks_coordinates(img.size[1], img.size[0], INITIAL_RANGE_BLOCK_SIZE)
        transforamtions_list = []

        for range_block_size in RANGE_BLOCKS_SIZES:
            if range_block_coordinates.size == 0:
                break
            if range_block_size in programs_dict:
                program = programs_dict[range_block_size]
            else:
                program = cl.Program(ctx, kernel).build(options=['-D', 'range_block_size={0}'.format(range_block_size)])
                programs_dict[range_block_size] = program

            range_coordinates = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                          hostbuf=range_block_coordinates)

            transformations = np.zeros((len(range_block_coordinates),), transfomation_struct)
            out_transformations_buffer = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, transformations.nbytes)

            program.find_matches.set_scalar_arg_dtypes([None, None, None, np.int32, np.int32, None])
            find_matches_event = program.find_matches(queue, (len(range_block_coordinates), 1), None, init_image,
                                                      downsampled_image, range_coordinates, np.int32(img.size[1]),
                                                      np.int32(img.size[0]),
                                                      out_transformations_buffer)

            got_transformations_event = cl.enqueue_copy(queue, transformations, out_transformations_buffer,
                                                        wait_for=[find_matches_event])
            got_transformations_event.wait()
            range_block_coordinates = []
            keeped_range_blocks = []
            for tr in transformations:
                if tr[7] < TRESHOLD:
                    transforamtions_list.append(Transformation.from_struct(range_block_size, range_block_size*2, tr))
                else:
                    keeped_range_blocks.append(tr)
                    range_block_coordinates.extend(expand_coordinates(tr[2], tr[3], range_block_size))
            range_block_coordinates = np.array(range_block_coordinates)
        else:
            transforamtions_list.extend([Transformation.from_struct(range_block_size, range_block_size*2, tr)
                                         for tr in keeped_range_blocks])
        channels_transformations.append(transforamtions_list)
    print time.time() - start_time
    return channels_transformations
