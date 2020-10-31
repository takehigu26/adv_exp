import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

german_sensitive_features_dict = {"gender": 8, "age": 12}
def get_german(sensitive_feature_name="gender", remove_z=False, **kwargs):
    df = pd.read_csv("datasets/german/german.data-numeric", header=None, delim_whitespace=True)

    # change label from (1,2) into (1,0)
    label_idx = len(list(df)) - 1
    df[label_idx] = df[label_idx].map({2: 0, 1: 1})

    # create train data
    M = df.values
    Xtr, Xts, ytr, yts = train_test_split(M[:, :-1], M[:, -1], test_size=0.2, random_state=42)
    return Xtr, Xts, ytr, yts

def get_my_german(return_df=False, is_numeric=True, **kwargs):
    if is_numeric: df = pd.read_csv("datasets/german/my_german_numeric.csv")
    else: df = pd.read_csv("datasets/german/my_german.csv")

    # create train data
    M = df.values
    Xtr, Xts, ytr, yts = train_test_split(M[:, :-1], M[:, -1], test_size=0.2, random_state=42)
    if return_df:
        return df
    else:
        return Xtr, Xts, ytr, yts

def get_titanic(return_df=False, **kwargs):
    df = pd.read_csv('datasets/titanic/my_titanic.csv')
    df.drop(columns=['Unnamed: 0'], inplace=True)
    cols = list(df.columns)
    cols.remove('Survived')
    cols = cols + ['Survived']
    df = df.reindex(columns=cols)
    M = df.values
    Xtr, Xts, ytr, yts = train_test_split(M[:, :-1], M[:, -1], test_size=0.2, random_state=42)
    to_int = lambda z: np.array(z, np.int64)
    Xtr, Xts, ytr, yts = to_int(Xtr), to_int(Xts), to_int(ytr), to_int(yts)
    if return_df:
        return df
    else:
        return Xtr, Xts, ytr, yts

def get_adult(sensitive_feature_name, scale=True, remove_z=False, verbose=0, **kwargs):
    file_path = "datasets/adult/"

    if scale:
        file_name = "adult.npz"
        arr_holder = np.load(os.path.join(file_path, file_name))
        fit_scale = arr_holder[arr_holder.files[0]]
        M = fit_scale
    else:
        file_name = "adult.data"
        df = pd.read_csv(os.path.join(file_path, file_name), sep=",", header=None)
        d_label = defaultdict(LabelEncoder)
        fit = df.apply(lambda x: d_label[x.name].fit_transform(x) if x.dtype == np.dtype('O') else x)
        M = fit.values

    X_train, X_test, y_train, y_test = train_test_split(M[:, :-1], M[:, -1], test_size=0.2, random_state=42)
    z_idx = get_z_idx(sensitive_feature_name, adult_sensitive_features_dict)
    Xtr, Xts, Ztr, Zts, ytr, yts = extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z)
    return Xtr, Xts, ytr, yts, Ztr, Zts

# utils
dataset_fs = [get_german, get_adult]# ''', get_bank, get_compass''']

def prep_data(Xtr, Xts, ytr, yts, verbose=1):
    from tensorflow.keras.utils import to_categorical
    X_train = np.hstack([Xtr])
    Y_train = to_categorical(ytr)
    X_test = np.hstack([Xts])
    Y_test = to_categorical(yts)
    for x in [X_train, X_test, Y_train, Y_test]:
        if verbose > 1:
            print(x.shape)
    return X_test, X_train, Y_test, Y_train

# sub functions
def extract_z(X_test, X_train, y_test, y_train, z_idx, remove_z):
    if remove_z:
        ix = np.delete(np.arange(X_train.shape[1]), z_idx)
    else:
        ix = np.arange(X_train.shape[1])
    Xtr = X_train[:, ix]
    Ztr = X_train[:, z_idx].reshape(-1, 1)
    Xts = X_test[:, ix]
    Zts = X_test[:, z_idx].reshape(-1, 1)
    ytr = y_train
    yts = y_test
    return Xtr, Xts, Ztr, Zts, ytr, yts

def get_z_idx(sensitive_feature_name, sensitive_features_dict):
    z_idx = sensitive_features_dict.get(sensitive_feature_name, None)
    if z_idx is None:
        print("Feature {} not recognized".format(sensitive_feature_name))
        z_idx = 0
    return z_idx

    return X_test, X_train, Y_test, Y_train
