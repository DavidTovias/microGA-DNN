Complete implementation of thesis' code 

paradigm: OOP

load_data: input_params/[json file]
save_results: json file


Paralell method (threads with threading library)

dataset-id = [0 - 11]  

number of experiment = [0 - 9]  

RUN  

python main.py [dataset-id] microga.json [number of experiment]

   
   
Note:
input_params/microGa*

**WITHOUT FEATURE SELECTION (OPTIMIZE JUST L1, L2 AND LEARNING REATE)**
"method_name": "MicroGa",
"ind_representation": "MicroGa-binary",
"obj_fun": "acc"  or  "acc-pneurons"


**FEATURE SELECTION (MRMR or without MRMR)**
"method_name": "MicroGa-FS",
"ind_representation": "MicroGa-binary-FS",
"obj_fun": "acc_MRMR"  or  "acc-pneurons_MRMR"  or  "acc"  or "acc-pneurons"


**SORT BY RELEVANCE (MR)**
"method_name": "MicroGa-sort_MR",
"ind_representation": "MicroGa-binary_sort_MR-FS",
"obj_fun": "sort-MR_acc"  or  "sort-MR_acc-pneurons"


**SORT BY MIN_REDUNDANCY - MAX_RELEVANCE  (MRMR)**
"method_name": "MicroGa-sort_MRMR",
"ind_representation": "MicroGa-binary_sort_MRMR-FS",
"obj_fun": "sort-MRMR_acc"  or  "sort-MRMR_acc-pneurons"
