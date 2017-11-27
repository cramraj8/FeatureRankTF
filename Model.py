# -*- coding: utf-8 -*-
# @author = __ramraj__


from __future__ import absolute_import, division, print_function
import tensorflow as tf
slim = tf.contrib.slim


ACTIVATION_DICT = {'relu': tf.nn.relu,
                   'tanh': tf.nn.tanh,
                   'sigmoid': tf.nn.sigmoid}
INITIALIZER_DICT = {'random_normal': tf.random_normal_initializer(),
                    'truncated_normal': tf.truncated_normal_initializer(stddev=0.01),
                    'random_uniform': tf.random_uniform_initializer(),
                    'xavier': tf.contrib.layers.xavier_initializer(uniform=True),
                    'zeros': tf.zeros_initializer(),
                    'ones': tf.ones_initializer()}
REGULARIZER_DICT = {'l2': slim.l2_regularizer(0.0005),
                    'l1': slim.l1_regularizer(0.0005),
                    'elastic': slim.l1_l2_regularizer(0.0005, 0.0005)}


class Model(object):

    def __init__(self, **scope_arg):
        self._scope_arg = scope_arg

        self.WB_init = INITIALIZER_DICT.get(self._scope_arg['init'],
                                            tf.truncated_normal_initializer(stddev=0.01))
        self.WB_reg = REGULARIZER_DICT.get(self._scope_arg['reg'],
                                           slim.l2_regularizer(0.0005))
        self.do_train = self._scope_arg.get('do_train', True)

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=self.WB_init,
                            weights_regularizer=self.WB_reg,
                            biases_initializer=self.WB_init,
                            biases_regularizer=self.WB_reg,
                            trainable=self.do_train
                            ):
            pass

    def _full_zip(self, a, b):
        """Zip two lists upto the maximum length of any list by replicating the
        last occured element from the shorter list.

        """
        return zip(a, b) + [(x, b[-1])
                            for x in a[len(b):]] + [(a[-1], x)
                                                    for x in b[len(a):]]

    def build_graph(self, input, n_feat, user_arg):
        """Layerwise model building given the user-specifications.

        Last followed up specs will be used for further layers building.

        Parameters
        ==========
        input : numpy.ndarray
            A M * N array containing feature vectors.

        n_feat : integer
            Represents the number of columns in the input data.

        user_arg : lists of list
            Carries layer-wise specification.

        Return
        ======
        net : numpy.ndarray
            A M * N array containing latent feature.

        """

        self.layers = user_arg[0]
        self.dropouts = user_arg[1]
        self.activ_functs = user_arg[2]

        # INPUT LAYER
        self.net = slim.fully_connected(input, n_feat, scope='input_layer')

        # Custom zipping function call
        self._zipped_AB = self._full_zip(self.layers, self.dropouts)
        self._zipped_ABC = self._full_zip(self._zipped_AB, self.activ_functs)

        # HIDDEN LAYERS
        for [n_neurons, dropout], non_lin in self._zipped_ABC:
            self.net = slim.dropout(
                slim.fully_connected(self.net,
                                     n_neurons,
                                     activation_fn=ACTIVATION_DICT.get(non_lin,
                                                                       tf.nn.relu)),
                keep_prob=dropout)

        # OUTPUT LAYER
        self.net = slim.fully_connected(self.net, 1, scope='output_layer')

        return self.net
