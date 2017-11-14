import tensorflow as tf
import numpy as np


def build(inputs, num_labels, labels):
    model = tf.nn.relu(_conv_layer(
        inputs, 32, 9, 1, num_labels, labels))
    model = tf.nn.relu(_conv_layer(
        model, 64, 3, 2, num_labels, labels))
    model = tf.nn.relu(_conv_layer(
        model, 128, 3, 2, num_labels, labels))
    model = _residual_layer(model, 128, num_labels, labels)
    model = _residual_layer(model, 128, num_labels, labels)
    model = _residual_layer(model, 128, num_labels, labels)
    model = _residual_layer(model, 128, num_labels, labels)
    model = _residual_layer(model, 128, num_labels, labels)
    model = tf.nn.relu(_upsampling_layer(
        model, 64, num_labels, labels))
    model = tf.nn.relu(_upsampling_layer(
        model, 32, num_labels, labels))
    model = tf.nn.sigmoid(_conv_layer(
        model, 3, 9, 1, num_labels, labels))

    return model


def _conv_layer(inputs, out_filters, kernel_size, stride, num_labels, labels):
    pad_size = kernel_size // 2
    inputs = tf.pad(inputs, mode='REFLECT', paddings=[
                    [0, 0], [pad_size, pad_size], [pad_size, pad_size], [0, 0]])
    padding = 'valid'
    return _conditional_instance_normalization(
        tf.layers.conv2d(inputs, out_filters, kernel_size, strides=stride, padding=padding), out_filters, num_labels, labels)


def _conditional_instance_normalization(inputs, channels, num_labels, labels):
    mean, variance = tf.nn.moments(inputs, axes=[1, 2])
    mean = tf.expand_dims(tf.expand_dims(mean, axis=1), axis=1)
    variance = tf.expand_dims(tf.expand_dims(variance, axis=1), axis=1)
    gamma = tf.Variable(np.ones((num_labels, channels)
                                ).astype('float32'), name='gamma')
    beta = tf.Variable(np.zeros((num_labels, channels)
                                ).astype('float32'), name='beta')
    cond_gamma = tf.expand_dims(tf.expand_dims(
        tf.gather(gamma, labels), axis=1), axis=1)
    cond_beta = tf.expand_dims(tf.expand_dims(
        tf.gather(beta, labels), axis=1), axis=1)
    epsilon = 1e-3
    return (cond_gamma * (inputs - mean)) / ((variance + epsilon) ** 0.5) + cond_beta


def _upsampling_layer(inputs, channels, num_labels, labels, scale=2):
    input_shape = tf.shape(inputs)
    resized_images = tf.image.resize_images(inputs,
                                            [input_shape[1] * scale,
                                                input_shape[2] * scale],
                                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return _conv_layer(inputs=resized_images,
                       out_filters=channels,
                       kernel_size=3,
                       stride=1,
                       num_labels=num_labels,
                       labels=labels)


def _residual_layer(inputs, channels, num_labels, labels):
    conv_1 = _conv_layer(inputs=inputs,
                         out_filters=channels,
                         kernel_size=3,
                         stride=1,
                         num_labels=num_labels,
                         labels=labels)
    conv_1 = tf.nn.relu(conv_1)
    conv_2 = _conv_layer(inputs=conv_1,
                         out_filters=channels,
                         kernel_size=3,
                         stride=1,
                         num_labels=num_labels,
                         labels=labels)
    return inputs + conv_2
