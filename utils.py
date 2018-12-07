import numpy as np
import tensorflow as tf


def softmax(inpt):
    e_x = np.exp(inpt - np.max(inpt))
    return e_x / e_x.sum()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def gauss_neg_log_pi(action_taken, mean, std, log_std):
    return 0.5 * tf.reduce_sum(tf.square((action_taken - mean) / std), axis=-1)\
           + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(action_taken)[-1])\
           + tf.reduce_sum(log_std, axis=-1)


def gauss_neg_log_pi_per_dim(action_taken, mean, std, log_std):
    return 0.5 * tf.square((action_taken - mean) / std) + 0.5 * np.log(2.0 * np.pi) + log_std


def gauss_likelihood(sample, mean, std):
    return tf.exp(-0.5 * tf.square((sample - mean) / std)) / (std * np.sqrt(2.0 * np.pi))


def expand_dim_cos(inpt, target_dimensionality):
    tf.assert_less_equal(inpt, 1.0)
    tf.assert_greater_equal(inpt, 0.0)
    dimensionality_expansion = target_dimensionality // int(inpt.shape[-1])
    if dimensionality_expansion <= 1:
        return inpt
    embedding_indices = tf.reshape(tf.range(1, dimensionality_expansion + 1, dtype=tf.float32),
                                   [1] * len(inpt.shape) + [dimensionality_expansion])
    target_shape = tf.concat([inpt.shape[:-1], [dimensionality_expansion * int(inpt.shape[-1])]], axis=0)
    return tf.reshape(tf.cos(np.pi * tf.expand_dims(inpt, axis=-1) * embedding_indices), target_shape)


def huber_loss(inpt, kappa=1.0):
    return tf.where(tf.abs(inpt) < kappa, tf.square(inpt) * 0.5, kappa * (tf.abs(inpt) - 0.5 * kappa))


def quantile_loss(quantile, delta, huber):
    quantile_scale = tf.where(delta < 0.0, 1.0 - quantile, quantile)
    tf.assert_greater_equal(quantile_scale, 0.0)
    tf.assert_less_equal(quantile_scale, 1.0)
    if huber:
        return quantile_scale * huber_loss(delta, 0.01)
    else:
        return quantile_scale * tf.abs(delta)