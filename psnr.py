import Image
import numpy as np


def get_psnr(first_file, second_file):
    first_image = Image.open(first_file)
    second_image = Image.open(second_file)

    mse = 0
    for f_ch, s_ch in zip(first_image.split(), second_image.split()):
        mse += np.sum((np.array(f_ch.getdata()) - np.array(s_ch.getdata()))**2)
    size = first_image.size
    mse /= (size[0]*size[1]*3)
    return 10*np.log10((255**2)/mse)

if __name__ == '__main__':
    print get_psnr('nature.bmp', 'nature_quadtree_16_100.bmp')