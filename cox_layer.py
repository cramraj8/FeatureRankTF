import tensorflow as tf
import numpy as np


def cost_function_censored(predictions, at_risk_label, censored):

    with tf.name_scope('loss_function'):

        max_preds = tf.reduce_max(predictions, axis=0)
        factorized_preds = predictions - max_preds
        exp = tf.exp(factorized_preds)
        partial_sum = tf.cumsum(exp, reverse=True)
        partial_sum_at_risk = tf.gather(partial_sum, at_risk_label)
        log_at_risk = tf.log(partial_sum_at_risk) + max_preds
        diff = tf.subtract(predictions, log_at_risk)
        diff_uncensored = tf.reshape(diff, [-1]) * (1 - censored)
        return -tf.reduce_sum(diff_uncensored)


def cost_function_observed(predictions, at_risk_label, observed):

    with tf.name_scope('loss_function'):

        max_preds = tf.reduce_max(predictions, axis=0)
        factorized_preds = predictions - max_preds
        exp = tf.exp(factorized_preds)
        partial_sum = tf.cumsum(exp, reverse=True)
        partial_sum_at_risk = tf.gather(partial_sum, at_risk_label)
        log_at_risk = tf.log(partial_sum_at_risk) + max_preds
        diff = tf.subtract(predictions, log_at_risk)
        diff_uncensored = tf.reshape(diff, [-1]) * observed
        return -tf.reduce_sum(diff_uncensored)


if __name__ == '__main__':
    cost()
