import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer

from alphagan_class import AlphaGAN
from bigan import BIGAN
from util import dump_column_transformers, load_column_transformers, split_data, preprocess_data

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # cc_df = pd.read_csv(sys.argv[1]) # Credit Card DataFrame
        # cc_df.drop(columns=['Time', 'Amount'])

        # split_data('./creditcard.csv', './data')

        train_df = pd.read_csv(sys.argv[1])
        preprocess_data(train_df, './data/ranges.csv')

        X_train = train_df.to_numpy()

        ag = AlphaGAN()
        ag.train(X_train=X_train, epochs=4000, batch_size=32)

        # ag = BIGAN()
        # ag.train(X_train=X_train, epochs=4000, batch_size=64)

        # test_normal_df = pd.read_csv('./data/test_set_normal.csv').head(10)
        # preprocess_data(test_normal_df, './data/ranges.csv')
        #
        # test_abnomal_df = pd.read_csv('./data/test_set_abnomal.csv').head(10)
        # preprocess_data(test_abnomal_df, './data/ranges.csv')
        #
        # X_test1 = test_normal_df.to_numpy()
        # X_test2 = test_abnomal_df.to_numpy()
        #
        # Y1 = ag.discriminator.predict(X_test1)
        # print(Y1)
        #
        # Y2 = ag.discriminator.predict(X_test2)
        # print(Y2)
