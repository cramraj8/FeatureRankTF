# -*- coding: utf-8 -*-
# @author='Ramraj'


from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from SurvivalAnalysis import SurvivalAnalysis
import normalize
from bayes_opt import BayesianOptimization as bayesopt
from Model import Model


# ******************************************************************************
BETA = 0.0001
TRAINING_EPOCHS = 8
BATCH_SIZE = 100
DISPLAY_STEP = 100

# LEARN_RATE = 0.0002
INITIAL_LEARNING_RATE = 0.0002
LEARNING_RATE_DECAY_FACTOR = 0.7
NUM_OF_EPOCHS_BEFORE_DECAY = 1000

# ============ Network Parameters ============
DEPTH = 3
MAXWIDTH = 500
N_CLASSES = 1
# ******************************************************************************

SCOPE_ARG = {'init': "truncated_norm", 'reg': "l2", 'do_train': True}
INPUT_ARG = [[300, 500, 450, 200],
             [0.5, 0.4],
             ["relu", "tanh", "sigmoid"]]
# ******************************************************************************


def data_provider(data_file='./survivalData.csv'):

    data_feed = pd.read_csv(data_file, skiprows=None, header=None)

    feature_names = data_feed.loc[0, 3:]

    data_feed.columns = data_feed.iloc[0]
    data_feed = data_feed[1:]

    survival = data_feed['Survival Time']
    censored = data_feed['Censored Status']

    feature_matrix = data_feed.iloc[0:, 3:]

    survival = np.asarray(survival, dtype=np.float32)
    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    censored = np.asarray(censored, dtype=np.float32)
    feature_names = list(feature_names)

    return feature_matrix, survival, censored, feature_names


def cox_layer_cost_function(predictions, at_risk_label, censored):

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


# ******************************************************************************
# ******************************************************************************

def train():
    r"""Function for regular training.

    """

    with tf.Graph().as_default() as graph:
        logging.set_verbosity(tf.logging.INFO)

        data_x, data_y, data_c, feat_names = data_provider('./adrcSurvivalNetModel1.csv')
        data_x, data_y, data_c = shuffle(data_x, data_y, data_c, random_state=1)
        data_x = normalize.z_score_normalization(data_x)

        X = data_x
        C = data_c
        T = data_y

        fold = int(len(X) / 10)
        train_set = {}
        test_set = {}
        final_set = {}

        sa = SurvivalAnalysis()
        train_set['X'], train_set['T'], train_set['C'], train_set['A'] = sa.calc_at_risk(X[0:fold * 6, ], T[0:fold * 6], C[0:fold * 6]);

        print('Shape of X : ', train_set['X'].shape)
        print('Shape of T : ', train_set['T'].shape)
        print('Shape of C : ', train_set['C'].shape)

        total_observations = train_set['X'].shape[0]
        input_features = train_set['X'].shape[1]
        observed = 1 - train_set['C']

        x = tf.placeholder("float", [None, input_features], name='features')
        c = tf.placeholder("float", [None], name='censored')
        a = tf.placeholder(tf.int32, [None], name='at_risk')

        ckpt_dir = './log/'
        if not tf.gfile.Exists(ckpt_dir):
            tf.gfile.MakeDirs(ckpt_dir)

        num_batches_per_epoch = total_observations / BATCH_SIZE
        num_steps_per_epoch = num_batches_per_epoch
        decay_steps = int(NUM_OF_EPOCHS_BEFORE_DECAY * num_steps_per_epoch)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        model = Model(**SCOPE_ARG)
        pred = model.build_graph(x, input_features, INPUT_ARG)

        lr = tf.train.exponential_decay(learning_rate=INITIAL_LEARNING_RATE,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate=LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)

        # ********************** LOSS && OPTIMIZE *********************************
        loss = cox_layer_cost_function(pred, a, c)
        # **************************************************************************
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, global_step=global_step)

        FeatRisks = tf.gradients(pred, x, name='Feature_Risks')
        # **************************************************************************
        # Launch the graph
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(TRAINING_EPOCHS):

                featrisks, _, avg_cost, prediction = sess.run([FeatRisks, optimizer,
                                                               loss,
                                                               pred],
                                                              feed_dict={x:
                                                                         train_set['X'],
                                                                         c:
                                                                         train_set['C'],
                                                                         a:
                                                                         train_set['A']})

                print('cost : ', avg_cost)

                featrisks = np.asarray(featrisks, np.float32)
                featrisks = np.squeeze(featrisks)

                R = np.array(abs(featrisks))
                R_avg = np.nanmean(R, axis=0)
                R_avg_norm = np.divide((R_avg - np.nanmean(R_avg)), np.nanstd(R_avg))

                Order = np.argsort(R_avg_norm)
                File = './Risk_rank.rnk'
                # open rnk file
                try:
                    Rnk = open(File, 'w')
                except IOError:
                    print("Cannot create file ", File)

                # write contents to file
                for i in Order:
                    name = '%s : \t %s \n' % (str(feat_names[i]), str(R_avg_norm[i]))
                    Rnk.write(name)

                # close file
                Rnk.close()

            print("Training Finished!")
            print("Rank file saved!")

            # Getting concordance index
            # c = sa.c_index(prediction, train_set['T'], train_set['C'])
            c = sa.c_index(prediction[:1000],
                           train_set['T'][:1000],
                           train_set['C'][:1000])
            print("c_index = {}".format(c))


train()
