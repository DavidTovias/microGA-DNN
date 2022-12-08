################## 
# 
# IndividualMicroGA Class 
#   
##################


import numpy as np
import copy
from obj_fun.obj_fun import objFun
from abstract.abs_individuals import IIndividual

class IndividualMicroGA_SORT_MRMR(IIndividual):

    def __init__(self, params):
        self.params = copy.deepcopy(params)
        # total dimensions of individual
        self.dim = self.params['dim'] # + self.params['total_features']
        # setup params
        self.setup_params()
        # number of features selected
        self.num_of_FS = 0
        # fitness of individual
        self.fitness = None
        self.acc = None
        self.pneurons = None
        # binary chromosome (genotype, bindim-dimensional)
        self.chromosome = np.random.randint(2,size=(self.params['bindim']),dtype=np.uint8)
        # real values (fenotype, vector dim-dimensional)
        self.xreal = np.zeros(self.dim)
        # history of the train and test (at the end)
        self.history_train = None
        self.eval_test = None
    
    # to print the values in ind
    def __str__(self):
        obj = "IndividualMicroGAFS object\n"
        self.xreal[-1] = int(self.xreal[-1]) # if not 2048 (if column(s) were deleted), could be a float number so cast to int
        obj += "chrom:"+str(self.chromosome) + "\nfitness:" + str(self.fitness) + "\nx:" + str(self.xreal)+"\n"
        obj += "accuracy:"+str(self.acc) + "\npneurons:"+str(self.pneurons)+"\n"
        return obj

    # set up the params needed to create an individual
    def setup_params(self):
        #print("En setup_params")
        total_features = self.params['total_features']

        # Min and max boundaries of the problem variables
        lmin = self.params['mn']
        lmax = self.params['mx']

        # append
        lmin.append(1)
        lmax.append(total_features)
        #print("LMIN: ", lmin)
        #print("LMAX:", lmax)
        
        precision = self.params['precision']
        precision.append(0)
        
        #print("precision", precision)
        
        # add dimension to dim
        self.dim += 1
        # Bits per dimension of the problem
        nbits = np.zeros(self.dim,dtype=np.int32)
        for i in range(self.dim):
            nbits[i] = np.floor(np.log2((lmax[i] - lmin[i])*10**precision[i])) + 1
        # Chromosome size
        bindim = np.sum(nbits)

        # add nbits and bindim to params
        self.params.update({'nbits':nbits, 'bindim':bindim, 'lmin':lmin, 'lmax':lmax})

        #print(bindim)
        #print(self.dim)
        
        
        
    
    # Decode binary individual into real    
    def decode(self):
        dim, nbits = self.dim, self.params['nbits']
        lmin, lmax = self.params['lmin'], self.params['lmax']
        
        index = np.concatenate(([0],np.cumsum(nbits)),axis=0)

        for i in range(0,dim):
            num = np.arange(0,nbits[i]) 
            start = index[i]
            finish = index[i+1]
            bin_ = self.chromosome[start:finish]
            integer = np.sum(bin_*(2**num))
            self.xreal[i] = (integer/(2**nbits[i]-1))*(lmax[i] - lmin[i]) + lmin[i]
            

    # Evaluate on objective function
    def evaluate(self, data):
        # decode binary chromosome into real (fenotype)
        self.decode()

        # update self.num_of_FS
        self.num_of_FS = int(self.xreal[-1])
    
        self.fitness, self.acc, self.pneurons = objFun(self.xreal, data, self.params)





    