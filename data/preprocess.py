import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from data.load_data import get_params, get_paths_and_names

# globals
path_datasets = None
dataset = None

# get index of feature that all elements are equal
def get_featureIndex(X):
    index = []
    for i in range(len(X[0])):
        if np.max(X[:,i]) == np.min(X[:,i]):
            index.append(i)

    return index

# normalize between 0,1
def normalize_data(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    X = scaler.transform(data)

    return X


def get_data(dataset_id):
    global path_datasets, dataset

    path_datasets, names = get_paths_and_names()
    # get the dataset name
    dataset = names[dataset_id]

    # get data from path
    data = pd.read_csv(path_datasets[dataset], header=None)
    # convert to numpy
    data = np.array(data)
    
    # get variables and class labels
    X = data[:,0:-1]
    Y = data[:,-1]
    del data

    return X, Y, dataset


def delete_columns(Xtr, Xtt, obj_fun):

    # if is a MRMR or MR objective function
    if "MR" in obj_fun:
        # get index of features (with equals values min==max)
        index = get_featureIndex(Xtr)
        print("\nindex (columns to delete):", index)
        if len(index) == 0:
            print("No column removed\n")
        else:
            # delete columns
            Xtr = np.delete(Xtr,index,axis=1)
            Xtt = np.delete(Xtt,index,axis=1)
            print("\n" + dataset + " (Xtr-Training, features) (new shape) \n", np.shape(Xtr), "\n")

    print("Xtr:", len(Xtr), "for training")
    print("Xtt:", len(Xtt), "for final test")

    
    return Xtr, Xtt
