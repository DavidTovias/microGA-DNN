import json

# read a json file
f = open('microGa_l1l2eta_25.json')
params = json.load(f)

f.close()

print((params['folds']))