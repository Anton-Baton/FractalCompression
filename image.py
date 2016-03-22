__author__ = 'anton'


from PIL import Image
import numpy as np


class ImageData(object):
    def __init__(self, width, height, image_data, mode):
        self.width = width
        self.height = height
        self.image_data = image_data
        self.channels = len(image_data)
        self.mode = mode


def load_image(filename):
    image = Image.open(filename)
    r, g, b = image.split()
    r_data = np.reshape(r.getdata(), image.size)
    g_data = np.reshape(g.getdata(), image.size)
    b_data = np.reshape(b.getdata(), image.size)
    image_data = np.array([r_data, g_data, b_data])
    img = ImageData(*image.size, image_data=image_data, mode=image.mode)
    return img


def save_image(filename, img):
    image = Image.fromarray(np.dstack(img.image_data), img.mode)
    image.save(filename)

