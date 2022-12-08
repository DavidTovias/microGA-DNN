###############
#
# Implements metaheuristic factory
# Call the metaheuristic to run
#
###############

from Micro_GA.Micro_GA import MicroGA
from Micro_GA.Micro_GA_Acc import MicroGA_ACC

# Returns a metaheuristic object
# Params: string name of method
class MetaheuristicFactory:
    @staticmethod
    def build_metaheuristic(params, data):
        method = params['method_name']
        obj_fun = params['obj_fun']
        method__ = (method=='MicroGa' or method=='MicroGa-FS' or method=='MicroGa-sort_MR' or method=='MicroGa-sort_MRMR')
        if method__ and (obj_fun=='acc-pneurons' or obj_fun=='acc-pneurons_MRMR' or obj_fun=='sort-MR_acc-pneurons' or obj_fun=='sort-MRMR_acc-pneurons' ):
            print(method + " object")
            return MicroGA(params, data)
        elif method__ and (obj_fun=='acc' or obj_fun=='acc_MRMR' or obj_fun=='sort-MR_acc' or obj_fun=='sort-MRMR_acc'):
            print("MicroGA_ACC Object")
            return MicroGA_ACC(params, data)
        else:
            print("NONE_TYPE (Metaheuristics factory)")
            print("bye ...")
            exit()