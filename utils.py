from os import listdir
from scipy.misc import imread, imresize, imsave
import numpy as np


def save_image(image, img_name):
    image = image * 255.0
    imsave(img_name + ".jpeg", image)


def fetch_images(data, batch_size, index, shape):
    data_list = []
    for i in range(batch_size):
        data_list.append(
            imresize(imread(data[index + i], mode='RGB').astype('float32'), shape))
    return np.array(data_list)


def load_styles(style_folder, style_size):
    image_list = [imread(style_folder + '/' + img_name,
                         mode='RGB').astype('float32')
                  for img_name in listdir(style_folder)]
    image_list = [imresize(x, float(style_size) / min(x.shape[0], x.shape[1]))
                  for x in image_list]
    return image_list


def create_data(content_folder):
    data = [content_folder + '/' +
            img_name for img_name in listdir(content_folder)]
    return data
