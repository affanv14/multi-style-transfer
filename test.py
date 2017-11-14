import tensorflow as tf
from os import listdir
from scipy.misc import imread
from net import build
from utils import save_image
import argparse


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-folder', dest='test_folder',
                        help='folder containing test data')
    parser.add_argument('--save-file', dest='check_file', default='modelcheck',
                        help='name of checkpoint file')
    parser.add_argument('--num-styles', dest='num_styles', default=32,
                        type=int, help='number of styles in the network')
    args = parser.parse_args()
    data = [imread(args.test_folder + '/' +
                   img_name) for img_name in listdir(args.test_folder)]
    image_style(data, args.check_file, args.num_styles)


def image_style(test_data, save_file, num_styles):
    label_data = [i for i in range(num_styles)]
    label_data.sort()
    with tf.Graph().as_default(), tf.Session() as sess:
        test_images = tf.placeholder(tf.float32, shape=(None, None, None, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(None,))
        stylized_images = build(test_images / 255.0, 32, labels_placeholder)

        saver = tf.train.Saver()
        saver.restore(sess, save_file)
        for i, img in enumerate(test_data):
            for lab in label_data:
                feed_dict = {test_images: img[None, :,
                                              :, :], labels_placeholder: [lab]}
                images = sess.run(
                    [stylized_images], feed_dict=feed_dict)[0]
                save_image(images[0], 'image{}style{}'.format(i + 1, lab + 1))


test()
