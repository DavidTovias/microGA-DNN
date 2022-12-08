import json
import numpy as np

def save_results(plots, evals_and_time, best_ind, params):
    # General Results
    fitness_curves = plots['fitness_plots']
    acc_curves = plots['acc_plots']
    pneurons_curves = plots['pneurons_plots']
    algorithm_time = evals_and_time['algorithm_time']
    EVALUATIONS = evals_and_time['EVALUATIONS']
    EVALUATIONS_RESET = evals_and_time['EVALUATIONS_RESET']
    general = {"fitness_plots": fitness_curves, "acc_plots":acc_curves, "pneurons_plots":pneurons_curves, "algorithm_time":algorithm_time}
    general.update({"EVALUATIONS":EVALUATIONS, "EVALUATIONS_RESET":EVALUATIONS_RESET})
    

    # if FS Scheme 
    method = params['method_name']
    method_FS = method.split('-')[-1]
    opt = params['optimize']

    # Feature selection with or without MRMR
    if method_FS == 'FS' or method_FS == 'MRMR':
        FS = np.where(best_ind.xreal[opt:] == 1)
        features_selected = FS[0].tolist()

    # Sort MRMR
    elif "sort_MRMR" in method:
        FS = int(best_ind.xreal[3]) # number of features selected
        print("type FS", type(FS), "  FS:", FS)
        total_features = np.array(params["MRMR_index"])
        features_selected = total_features[:FS]
        features_selected = features_selected.tolist()

        general.update({"num_of_FS" : params["num_of_FS"]})

    # Sort MR
    elif "sort_MR" in method:
        FS = int(best_ind.xreal[3]) # number of features selected
        print("type FS", type(FS), "  FS:", FS)
        total_features = np.array(params["index_MR"])
        features_selected = total_features[:FS]
        features_selected = features_selected.tolist()

        general.update({"num_of_FS" : params["num_of_FS"]})
    else:
        features_selected = 2048

    # if Learning rate is optimized. None, eta is optimized
    if params['eta'] == None:
        eta = best_ind.xreal[2]
    else:
        eta = params['eta']


    # Results of best individual
    # train
    acc_train = best_ind.history_train.history["accuracy"][-1]
    history_train = best_ind.history_train.history["accuracy"]
    history_loss = best_ind.history_train.history["loss"]
    # test
    loss_test, acc_test = best_ind.eval_test[0], best_ind.eval_test[1]

    ind_results = {"l1":best_ind.xreal[0], "l2":best_ind.xreal[1], "pneurons":best_ind.pneurons, "lr":eta }
    
    # save the number of features selected (if is the case)
    try:
        ind_results.update({"Num_of_features_selected": len(features_selected),"features_selected":features_selected})
    except:
        ind_results.update({"features_selected":features_selected})

    ind_results.update({"history_train":history_train, "history_loss":history_loss, "acc_train":acc_train})
    best_ind_time = evals_and_time['best_ind_time']
    ind_results.update({"acc_test":acc_test, "loss_test":loss_test, "best_ind_time":best_ind_time})

    RESULTS = {}
    RESULTS.update({"general":general, "ind_results":ind_results})
    name = params['method_name'] + '/'
    representation = params['ind_representation'] + '/'
    fun = params['obj_fun'] + '/'
    dataset = params['dataset']

    # save results json file
    run = params['run']
    out_file = open('results/'+name+representation+fun+dataset+'/'+dataset+'_run'+str(run)+'.json','w')
    json.dump(RESULTS, out_file, indent=4)
    out_file.close()
