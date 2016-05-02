__author__ = 'anton'

import image
import encoder
import decoder


if __name__ == '__main__':
    img = image.load_image('fairis.bmp')
    print img.width, img.height
    transformations = encoder.encode(img)
    print len(transformations), len(transformations[0])
    img_data = decoder.decode(10, img.width, img.height, transformations)
    print len(transformations)
    img = image.ImageData(img.width, img.height, img_data, img.mode)
    image.save_image('fairis_test.bmp', img)
