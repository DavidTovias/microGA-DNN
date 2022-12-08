################
#
# Get the paths to datasets, and load json file input
#
################

import json

# return params of the algorithm
# Params: name of input file (json file)
def get_params(json_file):
    f = open('input_params/'+str(json_file))
    # get params from json
    params = json.load(f)
    f.close()

    return params



# return paths and names of all the datasets
# Params: none
def get_paths_and_names():
    dataset_paths = {'flowers': '../../feature_vectors/data-CSV/flowers.csv',
                 'rps': '../../feature_vectors/data-CSV/rps.csv',
                 'plants': '../../feature_vectors/data-CSV/plants.csv',
                 'srsmas': '../../feature_vectors/data-CSV/srsmas.csv',
                 'cataract': '../../feature_vectors/data-CSV/cataract.csv',
                 'skincancer': '../../feature_vectors/data-CSV/skincancer.csv',
                 'leaves': '../../feature_vectors/data-CSV/leaves.csv',
                 'weather': '../../feature_vectors/data-CSV/weather.csv',
                 'MIT_IndoorScenes': '../../feature_vectors/data-CSV/MIT_IndoorScenes.csv',
                 'chessman': '../../feature_vectors/data-CSV/chessman.csv',
                 'covid-19': '../../feature_vectors/data-CSV/covid-19.csv',
                 'painting': '../../feature_vectors/data-CSV/painting.csv'
                }

    names = ['srsmas','rps','leaves','painting','cataract','plants',
         'skincancer','flowers','weather','MIT_IndoorScenes',
         'chessman','covid-19']

    return dataset_paths, names



