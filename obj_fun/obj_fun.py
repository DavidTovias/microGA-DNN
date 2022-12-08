
import copy
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

from MLP.mlp import MLP






#Globals
index_MR = []

# Objective function 
def objFun(x, data, params):
    global index_MR
    
    # Xtr -> 5 FCV
    folds = params['folds']
    skf2 = StratifiedKFold(n_splits=folds, random_state=3, shuffle=True)
    
    #(Xtr1, Xtt1)
    Xtr, Ytr, = data[0], data[1]
    del data

    
    # hidden layers
    l1 = x[0]
    l2 = x[1]

    # eta: learning rate
    # if None, eta is optimized 
    if params['eta'] == None:
        eta = x[2]
    else:
        eta = params['eta']


    method = params['method_name']

    # if method include feature selection (without MR/MRMR)
    #method = params['method_name'].split('-')[-1]
    method_Fs = method.split('-')[-1]
    if method_Fs == "FS":
        # variables to optimize, in addition to FS
        opt = params['optimize']
        # Features selected x[3:]
        index = x[opt:]
        FS = np.where(index == 1)
        FS = FS[0]
        Xtr = Xtr[:,FS]
        #Xtt = Xtt[:,FS]
    
    # if is a feature selection based on minRedundancy-maxRelevance (MRMR). sorted by MRMR
    elif "sort_MRMR" in method:
        index_MR = params["MRMR_index"]
        FS = int(x[3]) # num of features selected
        print("numero de features:", FS)
        FS = index_MR[:FS] # get first FS'th selected indexes
        #print("\nfeatues seleccionadas\n", FS)
        Xtr = Xtr[:,FS]
        #print("Xtr\n", Xtr)

    # if is a feature selection based on relevance (MR). sorted by relevance
    elif "sort_MR" in method:
        #print("AQUI MR")
        index_MR = get_index_MR(Xtr, Ytr)
        FS = int(x[3]) # num of features selected
        #print("numero de features:", FS)
        FS = index_MR[:FS] # get first FS'th selected indexes
        #print("\nfeatues seleccionadas\n", FS)
        Xtr = Xtr[:,FS]
        #print("Xtr\n", Xtr)
        
    

    # shape
    input_shape = np.shape(Xtr[0])
    # params of the model to build
    mlp_params = {'l1':l1, 'l2':l2, 'eta':eta, 'input_shape':input_shape, 'num_classes':params['num_classes']}
    
    epochs = params['epochs']
    batch_size = params['batch_size']
    ############################
    #k = 1
    print("FOLD Cross-Validation")
    acc_train = 0
    for tr, vd in skf2.split(Xtr, Ytr):
        # create model
        model = MLP.build_model(mlp_params)
        
        #k += 1
        # Training set 
        Xtr1 = Xtr[tr]
        Ytr1 = Ytr[tr]
        
        # Validation set
        Xvd1 = Xtr[vd]
        Yvd1 = Ytr[vd]
        
        # convert to tensors
        Xtr1 = tf.convert_to_tensor(Xtr1, dtype=tf.float32)
        Ytr1 = tf.convert_to_tensor(Ytr1)
        Xvd1 = tf.convert_to_tensor(Xvd1, dtype=tf.float32)
        Yvd1 = tf.convert_to_tensor(Yvd1)
        
        # Training the model
        valid = (Xvd1, Yvd1)
        history_train = model.fit(Xtr1,Ytr1, validation_data=valid, epochs=epochs, verbose=0, batch_size=batch_size, shuffle=True)

        acc_train += history_train.history['val_accuracy'][-1]
    ############################
    
    # mean accuracy on validation set
    acc = acc_train/folds

    # percentage of neurons
    pneurons = (l1 + l2) / 1024.
    
    # objective function
    fun = params['obj_fun']
    
    # penalization with a minimum of units per layer
    min_units = params['min_units']
    if l1 < min_units or l2 < min_units:  
        f = 0
    else:
        if fun == "acc":
            print("fun:", fun)
            f = acc
        elif fun == "acc-pneurons":
            print("fun:", fun)
            f = 0.5*acc + 0.5*(1-pneurons) #, acc, mc, nmc
        elif fun == "acc-pneurons_MRMR":
            print("fun:", fun)
            MRMR = mrmr(Xtr, Ytr)
            f = (acc + (1-pneurons) + MRMR) / 3.
        elif fun == "acc_MRMR":
            print("fun:", fun)
            MRMR = mrmr(Xtr, Ytr)
            f = (acc + MRMR) / 2.
        elif fun == "sort-MR_acc":
            print("fun:", fun)
            f = acc
        elif fun == "sort-MR_acc-pneurons":
            print("fun:", fun)
            f = 0.5*acc + 0.5*(1-pneurons)
        
        # MRMR
        elif fun == "sort-MRMR_acc":
            print("fun:", fun)
            f = acc
        elif fun == "sort-MRMR_acc-pneurons":
            print("fun:", fun)
            f = 0.5*acc + 0.5*(1-pneurons)
    
    return f,acc,pneurons


# Point biserial correlation coefficient 
def pointbiserial(xi,y):
    noty = np.invert(y)
    nk = np.sum(y)
    nl = np.sum(noty)
    muk = np.mean(xi[y])
    mul = np.mean(xi[noty])
    Sxi  = np.std(xi,ddof=1)
    
    r = ( (muk - mul) / Sxi ) * np.sqrt( nk*nl / (nk+nl)**2 )
    return r 

# Minimum redundancy maximum relevance
def mrmr(X,Y):
    _,d = np.shape(X)
    labels = np.unique(Y)
    c = len(labels)
    Fij = np.zeros((d,c))
    for i in range(0,d):
        xi = X[:,i]
        for j in range(0,c):
            yj = Y == labels[j]
            Fij[i,j] = pointbiserial(xi,yj)
    F = np.mean(np.abs(Fij),1)
    del Fij
    #print("antes de corrcoef")
    R = np.abs(np.corrcoef(X,rowvar=False))
    #print("despues de corrcoef")
    del X
    V = np.sum(F) / d 
    W = np.sum(np.sum(R)) / d**2
    del R
    MRMR = (1 + (V-W))/2
    
    return MRMR 


def get_index_MR(X,Y):
    global index_MR
    if len(index_MR) == 0:
        print("len index_MR", len(index_MR) )
        _,d = np.shape(X)
        labels = np.unique(Y)
        c = len(labels)
        Fij = np.zeros((d,c))
        for i in range(0,d):
            xi = X[:,i]
            for j in range(0,c):
                yj = Y == labels[j]
                Fij[i,j] = pointbiserial(xi,yj)
        F = np.mean(np.abs(Fij),1)
        F_sort = np.argsort(F)
        F_sort_index = list(F_sort)
        #print("F_SORT 1", F[F_sort_index] )
        F_sort_index.reverse()
        #print("\n\nF_sort 2", F[F_sort_index])
        return np.array(F_sort_index)
    #else:
    #    print("No entro. len index_MR", len(index_MR) )
    #    print(index_MR)
    
    return index_MR