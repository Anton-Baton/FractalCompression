__author__ = 'anton'


import image
import encoder
import decoder


if __name__ == '__main__':
    img = image.load_image('lena.bmp')
    print img.width, img.height
    transformations = encoder.encode(img)
    print len(transformations)
    img = decoder.decode(img.width, img.height, transformations)
    print len(transformations)
    image.save_image('test_lena.bmp', img)
