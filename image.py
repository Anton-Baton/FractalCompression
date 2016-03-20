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
    image_data = np.array(map(lambda x: list(x.getdata()), image.split()), dtype=np.uint8)
    img = ImageData(*image.size, image_data=image_data, mode=image.mode)
    return img


def save_image(filename, img):
    reshaped_data = np.reshape(zip(*img.image_data), (img.width, img.height, img.channels))
    image = Image.fromarray(reshaped_data, img.mode)
    image.save(filename)

