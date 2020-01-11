import sys
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer


# Usage : dump_column_transformers(sys.argv[1], './transformers.pkl')
def dump_column_transformers(csv_path, dump_path):
    df = pd.read_csv(csv_path)  # Credit Card DataFrame
    df.drop(columns=['Time', 'Class', 'Amount'], inplace=True)

    transformers = []
    for i in range(df.shape[1]):
        col = df.columns[i]
        scaler = StandardScaler()
        # scaler.fit(df[col].to_numpy().reshape(-1, 1))
        transformers.append((col, scaler, [i]))

    col_transformer = ColumnTransformer(transformers)
    col_transformer.fit_transform(df.to_numpy())

    file = open(dump_path, 'wb')
    pickle.dump(col_transformer, file)


def load_column_transformers(dump_path):
    file = open(dump_path, 'rb')
    return pickle.load(file)


def preprocess_data(df, ranges_csv_path):
    ranges_df = pd.read_csv(ranges_csv_path)
    # ranges_df.set_index(pd.Index(['min', 'max', 'mean', 'stdev']))
    df['Amount'] = np.log10(df['Amount'].values + 1)

    data_cols = list(df.columns)
    df[data_cols] = (df[data_cols] - ranges_df.loc[2, data_cols]) / ranges_df.loc[3, data_cols]


def split_data(csv_path, save_path, test_ratio=0.3):
    df = pd.read_csv(csv_path)  # Credit Card DataFrame
    df.drop(columns=['Time', 'Class'], inplace=True)

    df['Amount'] = np.log10(df['Amount'].values + 1)

    data_cols = list(df.columns)
    percentiles = pd.DataFrame(np.array([np.percentile(df[i], [0.1, 99.9]) for i in data_cols]).T,
                               columns=data_cols, index=['min', 'max'])
    percentile_means = \
        [[np.mean(df.loc[(df[i] > percentiles[i]['min']) & (df[i] < percentiles[i]['max']), i])]
         for i in data_cols]
    percentiles = percentiles.append(pd.DataFrame(np.array(percentile_means).T, columns=data_cols, index=['mean']))
    percentile_stds = \
        [[np.std(df.loc[(df[i] > percentiles[i]['min']) & (df[i] < percentiles[i]['max']), i])]
         for i in data_cols]
    percentiles = percentiles.append(pd.DataFrame(np.array(percentile_stds).T, columns=data_cols, index=['stdev']))
    percentiles.to_csv(os.path.join(save_path, 'ranges.csv'), index=False)

    # recover original data
    df = pd.read_csv(csv_path)

    # split data to train and test
    df_normal = df[df['Class'] == 0]
    df_normal.drop(columns=['Time', 'Class'], inplace=True)

    split = int(df_normal.shape[0] * (1 - test_ratio))
    normal_train_df, normal_test_df = df_normal.head(split), df_normal.tail(df_normal.shape[0] - split)
    normal_train_df.to_csv(os.path.join(save_path, 'train_set_normal.csv'), index=False)
    normal_test_df.to_csv(os.path.join(save_path, 'test_set_normal.csv'), index=False)

    df_anomal = df[df['Class'] == 1]
    df_anomal.drop(columns=['Time', 'Class'], inplace=True)
    df_anomal.to_csv(os.path.join(save_path, 'test_set_anomal.csv'), index=False)
