#####################
#
#  Main file
#  
#  argv => [1]: dataset_id
#          [2]: json_file
#          [3]: number of run/execution in [0,9], set the seed
#####################

import pandas as pd
import numpy as np
import sys
import time
from mrmr import mrmr_classif
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import train_test_split

from data.save_results import save_results
from factory.metaheuristics import MetaheuristicFactory
from data.preprocess import delete_columns, get_data, normalize_data
from data.load_data import get_params

#import tensorflow as tf




# argv
dataset_id = int(sys.argv[1])
json_file = sys.argv[2]
run = int(sys.argv[3])

# Get params
params = get_params(json_file)
obj_fun = params['obj_fun']
print("Method: ", params['method_name'])
print("Ind Representation:", params['ind_representation'].split("-")[1:])
print("Objective Function: ", obj_fun)

# Get X, Y, and dataset name
X, Y, dataset = get_data(dataset_id)

# Normalize Data
X = normalize_data(X)

# print dataset name and its atributes X(instances, features). Y[classes]
print("\n",dataset + " (X-Complete set, features) (shape)")
print(np.shape(X))
print(np.unique(Y))



# divide into train, valid and test
# random state (seed) in [1,2]
seed = 2
if run%2 == 0:
    seed = 1
skf1 = StratifiedKFold(n_splits=params['folds'], random_state=seed, shuffle=True)
# Get train and test sets
tr1 = []
tt1 = []
for tr, tt in skf1.split(X, Y):
    #print("len tr", len(tr), "  len tt", len(tt))
    tr1.append(tr)
    tt1.append(tt)

# num_eval in [0,4]
num_eval = run//2
tr = tr1[num_eval]
tt = tt1[num_eval]

# Training set (split again into 5-FCV to eval the individual)
Xtr = X[tr]
Ytr = Y[tr]

# FINAL Test set
Xtt = X[tt]
Ytt = Y[tt]


# Preprocess data (delete some columns/features?)
if "MR" in obj_fun:
    Xtr, Xtt = delete_columns(Xtr, Xtt, obj_fun=obj_fun)


#print("shape Xtr", np.shape(Xtr))
#print("shape Xtt", np.shape(Xtt))
print("Preprocess OK.\n")



# add more elements to params.
n = np.size(X,axis=0)          # All patterns  
d = np.size(Xtr,axis=1)        # Dimensions - without columns removed (as in MRMR) 
c = np.size(np.unique(Y))      # Classes 
#shape = np.shape(X[0])         # input shape
#print("shape: ", shape)



## if sort by MRMR
if "sort_MRMR" in params["method_name"]:
    X1 = pd.DataFrame(Xtr)
    Y1 = pd.Series(Ytr)
    idx = mrmr_classif(X=X1, y=Y1, K=d)
    params.update({"MRMR_index": idx})


opt = dim = len(params['mn'])  # variables to optimize

params.update({'num_patterns':n, 'total_features':d, 'num_classes':c, 'optimize':opt, 'dim':dim})

#print(params)
#exit()


# instance of method/algorithm to use
algorithm = MetaheuristicFactory.build_metaheuristic(params, (Xtr, Ytr) )
del X, Y, Xtr, Ytr


start = time.time()
# run algorithm
fitness_curves, acc_curves, pneurons_curves = algorithm.run()
algorithm_time = time.time() - start


start = time.time()
# evaluate best individual on test data
algorithm.evaluate_best((Xtt, Ytt))
# training time of best architecture
best_ind_time = time.time() - start

# General Results
plots = {"fitness_plots": fitness_curves, "acc_plots":acc_curves, "pneurons_plots":pneurons_curves}
evals_and_time = {"EVALUATIONS":algorithm.EVALUATIONS, "EVALUATIONS_RESET":algorithm.EVALUATIONS_RESET}
evals_and_time.update({"algorithm_time":algorithm_time, "best_ind_time":best_ind_time})
params.update({"run":run, "dataset":dataset})

# MRMR
if "sort_MRMR" in params["method_name"]:
    params.update({"num_of_FS" : algorithm.params["num_of_FS"]})

elif "sort_MR" in params["method_name"]:
    # MR MaxRelevance    
    params.update({"num_of_FS" : algorithm.params["num_of_FS"], "index_MR":algorithm.best_ind.params["index_MR"]})

# Save results in file
save_results(plots=plots, evals_and_time=evals_and_time, best_ind=algorithm.best_ind, params=params)


