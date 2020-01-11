import sys
import os
import pandas as pd

from util import load_column_transformers, preprocess_data
from alphagan_class import AlphaGAN
from keras.losses import MeanAbsoluteError
from bigan import BIGAN

import keras.backend as K
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    session = K.get_session()
    init = tf.global_variables_initializer()
    session.run(init)

    ag = AlphaGAN()
    ag.load_pretrained_models('./snapshots/3900_')

    test_normal_df = pd.read_csv('./data/test_set_normal.csv')
    preprocess_data(test_normal_df, './data/ranges.csv')

    test_abnomal_df = pd.read_csv('./data/test_set_abnomal.csv')
    preprocess_data(test_abnomal_df, './data/ranges.csv')

    X_1 = test_normal_df.to_numpy()
    X_2 = test_abnomal_df.to_numpy()

    Z_hat_1 = ag.encoder.predict(X_1)
    X_hat_1 = ag.generator.predict(Z_hat_1)

    Z_hat_2 = ag.encoder.predict(X_2)
    X_hat_2 = ag.generator.predict(Z_hat_2)

    rec_losses_normal = np.linalg.norm(np.subtract(X_1, X_hat_1), axis=1)
    rec_losses_fraud = np.linalg.norm(np.subtract(X_2, X_hat_2), axis=1)

    num = len(rec_losses_normal) + len(rec_losses_fraud)
    print('Number of test samples: %d' % num)

    THRESH = 9.25

    rec_losses_normal_correct = [loss for loss in rec_losses_normal if loss < THRESH]
    print('Precision of normal transactions: %1.2f%%(%d/%d)' % (len(rec_losses_normal_correct) * 100 / len(rec_losses_normal),
          len(rec_losses_normal_correct), len(rec_losses_normal)))

    rec_losses_fraud_correct = [loss for loss in rec_losses_fraud if loss > THRESH]
    print('Precision of fraud transactions: %1.2f%%(%d/%d)' % \
          (len(rec_losses_fraud_correct) * 100 / len(rec_losses_fraud), len(rec_losses_fraud_correct), len(rec_losses_fraud)))
