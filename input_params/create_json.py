import json


dict = {
    'gen':100,
    'npop': 4,
    'dim': 3,
    'mn' : [1, 1, 1e-6],
    'mx' : [512, 512, 1e-1],
    'precision' :  [0,0,6],
    'pc' : 0.9,
    'eta' : None,
    'treshold' : 0.05,
    'folds' : 5,
    'name' : 'microGa_l1l2eta_25'
    }


# create a json format
jsonString = json.dumps(dict, indent=4)
print(jsonString)