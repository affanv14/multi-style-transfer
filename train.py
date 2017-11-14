import argparse
from utils import *
import tensorflow as tf
from net import build
from vgg import Vgg16

CONTENT_LAYER = 'conv3_3'
STYLE_LAYERS = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-folder', dest='data_folder',
                        help='folder containing training data')
    parser.add_argument('--style-folder', dest='style_folder',
                        help='folder containing styles')
    parser.add_argument('--vgg-path', dest='vgg_path',
                        help='path of vgg weights file')
    parser.add_argument('--save-file', dest='check_file', default='modelcheck',
                        help='name of checkpoint file')
    parser.add_argument('--summary-folder', dest='summary_folder', default='trainsummary',
                        help='folder for generating summaries')
    parser.add_argument('--iters', dest='num_iterations', default=40000,
                        type=int, help='number of training iterations')
    parser.add_argument('--batch-size', dest='batch_size', default=4,
                        type=int, help='number of samples per batch')
    parser.add_argument('--style-weight', dest='style_weight', default=30,
                        type=int, help='weight of style loss')
    parser.add_argument('--content-weight', dest='content_weight', default=1,
                        type=int, help='weight of content loss')
    parser.add_argument('--quiet', dest='quiet', action='store_true',
                        help='avoid printing training info')
    parser.add_argument('--use-checkpoint', dest='use_checkpoint', action='store_true',
                        help='use checkpoint file to continue training')
    parser.add_argument('--image-size', dest='image_size', default=256,
                        type=int, help='size of training images')
    parser.add_argument('--style-size', dest='style_size', default=256,
                        type=int, help='size of smaller side of style image')
    parser.add_argument('--learning-rate', dest='lr', default=1e-3,
                        type=float, help='learning rate of style model')
    args = parser.parse_args()

    data = create_data(args.data_folder)
    style_images = load_styles(args.style_folder, args.style_size)
    train(data, style_images, args.vgg_path, args.check_file, args.summary_folder,
          args.num_iterations, args.image_size, args.lr, args.batch_size,
          args.style_weight, args.content_weight, args.quiet,
          args.use_checkpoint)


def train(data, style_images, vgg_path, check_file, summary_file,
          num_iterations, image_size, lr, batch_size, style_weight, content_weight,
          quiet, use_checkpoint):
    num_styles = len(style_images)
    data = data[:-(len(data) % batch_size)]
    num_samples = len(data)
    vgg = Vgg16(vgg16_npy_path=vgg_path)
    # Calculate gram matrices for style images
    style_output = get_style_gram(style_images, vgg)

    # Build graph
    with tf.Graph().as_default(), tf.Session() as sess:
        style_placeholder = {}
        for layer in STYLE_LAYERS:
            style_placeholder[layer] = tf.placeholder(
                tf.float32, shape=(None, None, None))
        image_placeholder = tf.placeholder(
            tf.float32, shape=(batch_size, 256, 256, 3))
        labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size,))
        # content image
        vgg.build(image_placeholder / 255.0)
        original_content_output = getattr(vgg, CONTENT_LAYER)

        # generated image
        stylized_images = build(image_placeholder / 255.0, num_styles,
                                labels_placeholder)
        vgg.build(stylized_images)
        stylized_content_output = getattr(vgg, CONTENT_LAYER)
        # Calculate losses
        content_loss = tf.reduce_mean(tf.square(
            original_content_output - stylized_content_output))
        style_loss = get_style_loss(vgg, style_placeholder) * .25
        total_loss = style_weight * style_loss + content_weight * content_loss
        # Optimizer
        global_step = tf.Variable(0, name='global_step', trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        train_op = opt.minimize(total_loss, global_step=global_step)

        # Create summaries
        tf.summary.scalar('content_loss', content_loss)
        tf.summary.scalar('style_loss', style_loss)
        tf.summary.scalar('total_loss', total_loss)
        merged_summary = tf.summary.merge_all()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        if use_checkpoint:
            saver.restore(sess, check_file)
        writer = tf.summary.FileWriter(summary_file, sess.graph)

        image_shape = (image_size, image_size, 3)
        for iteration in range(num_iterations):
            imgs = fetch_images(data, batch_size,
                                (iteration * batch_size) % num_samples, image_shape)
            label_data = [iteration % num_styles] * batch_size
            feed_dict = {image_placeholder: imgs,
                         labels_placeholder: label_data}
            feed_dict.update(
                {style_placeholder[layer]: style_output[iteration % num_styles][layer] for layer in STYLE_LAYERS})

            _, step, tloss, sloss, closs, summary = sess.run(
                [train_op, global_step, total_loss, style_loss, content_loss, merged_summary], feed_dict=feed_dict)
            writer.add_summary(
                summary, step)

            if not quiet:
                print 'iteration {} style loss {} content loss {} total loss {}'.format(
                    iteration, sloss, closs, tloss)

            if iteration % 100 == 99:
                saver.save(sess, check_file)


def get_style_loss(vgg, style_gram):
    style_loss = 0
    for layer_name in STYLE_LAYERS:
        layer = getattr(vgg, layer_name)
        shape = tf.shape(layer)
        layer = tf.reshape(layer, (shape[0], -1, shape[3]))
        gram_output = tf.matmul(layer, layer, transpose_a=True) / \
            (tf.cast(shape[1] * shape[2] * shape[3], tf.float32))
        style_loss += tf.reduce_mean(
            tf.square(gram_output - style_gram[layer_name]))
    return style_loss


# try to make more efficient
def get_style_gram(style_images, vgg_model):
    style_list = []

    with tf.Graph().as_default(), tf.Session() as sess:

        styles_placeholder = tf.placeholder(
            tf.float32, shape=(1, None, None, 3))
        vgg_model.build(styles_placeholder / 255.0)
        for i, style in enumerate(style_images):
            style_list.append(dict())
            feed_dict = {styles_placeholder: style[None, :, :, :]}
            for layer_name in STYLE_LAYERS:
                layer = getattr(vgg_model, layer_name)
                shape = tf.shape(layer)
                layer = tf.reshape(layer, (1, -1, shape[3]))
                gram_output = tf.matmul(
                    layer, layer, transpose_a=True) / (tf.cast(shape[1] * shape[2] * shape[3], tf.float32))
                style_list[i][layer_name] = gram_output.eval(
                    feed_dict=feed_dict)

    return style_list


main()
