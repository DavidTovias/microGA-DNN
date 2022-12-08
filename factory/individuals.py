###############
#
# Implements individual factory
# Call the individual to evaluate
#
###############

from Micro_GA.individual_MicroGA import IndividualMicroGA
from Micro_GA.individual_MicroGA_FS import IndividualMicroGAFS
from Micro_GA.individual_MicroGA_sort_MR import IndividualMicroGA_SORT_MR
from Micro_GA.individual_MicroGA_sort_MRMR import IndividualMicroGA_SORT_MRMR

# Returns a metaheuristic object
# Params: string name of method
class IndividualFactory:
    @staticmethod
    def build_individual(params):
        ind_repr = params['ind_representation']
        if ind_repr == 'MicroGa-binary':
            return IndividualMicroGA(params)
        elif ind_repr == 'MicroGa-binary-FS':
            return IndividualMicroGAFS(params)
        elif ind_repr == 'MicroGa-binary_sort_MR-FS':
            return IndividualMicroGA_SORT_MR(params)
        elif ind_repr == 'MicroGa-binary_sort_MRMR-FS':
            return IndividualMicroGA_SORT_MRMR(params)
        else:
            print("NONE_TYPE (Individual factory)")
            print("bye ...")
            exit()