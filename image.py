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
    image_data = np.array(map(lambda x: np.reshape(x.getdata(), image.size), image.split()))
    img = ImageData(*image.size, image_data=image_data, mode=image.mode)
    return img


def save_image(filename, img):
    #data = zip(*img.image_data)

    #if img.channels == 1:
    #    reshaped_data = np.reshape(data, (img.height, img.width))
    #else:
    #    reshaped_data = np.reshape(data, (img.width, img.height, img.channels))
    bands = map(lambda ch: Image.fromarray(ch, 'L'), img.image_data)
    #image = Image.fromarray(reshaped_data, img.mode)
    image = Image.merge(img.mode, bands)
    image.save(filename)

