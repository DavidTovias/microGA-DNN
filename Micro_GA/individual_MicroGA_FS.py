################## 
# 
# IndividualMicroGA Class 
#   
##################


import numpy as np
from obj_fun.obj_fun import objFun
from abstract.abs_individuals import IIndividual

class IndividualMicroGAFS(IIndividual):

    def __init__(self, params):
        self.params = params
        # total dimensions of individual
        self.dim = self.params['dim'] + self.params['total_features']
        # setup params
        self.setup_params()
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
        obj += "chrom:"+str(self.chromosome) + "\nfitness:" + str(self.fitness) + "\nx:" + str(self.xreal)+"\n"
        obj += "accuracy:"+str(self.acc) + "\npneurons:"+str(self.pneurons)+"\n"
        return obj

    # set up the params needed to create an individual
    def setup_params(self):
        # Min and max boundaries of the problem variables
        """if np.size(self.params['mn']) == 1:
            lmin = np.zeros(self.params['dim'])
            lmax = np.zeros(self.params['dim']) 

            for i in range(0,self.params['dim']):
                lmin[i] = self.params['mn']
                lmax[i] = self.params['mx']
        else: 
            lmin = self.params['mn']
            lmax = self.params['mx']"""
        
        lmin = self.params['mn']
        lmax = self.params['mx']

        total_features = self.params['total_features']
        
        min_ = np.zeros(total_features)
        lmin = np.hstack((lmin,min_))
        
        max_ = np.ones(total_features)
        lmax = np.hstack((lmax,max_))
        
        precision = self.params['precision']
        prec = np.zeros(total_features)
        precision = np.hstack((precision, prec))

        # Bits per dimension of the problem
        nbits = np.zeros(self.dim,dtype=np.int32)
        for i in range(0,self.dim):
            nbits[i] = np.floor(np.log2((lmax[i] - lmin[i])*10**precision[i])) + 1
        # Chromosome size
        bindim = np.sum(nbits)

        # add nbits and bindim to params
        self.params.update({'nbits':nbits, 'bindim':bindim, 'lmin':lmin, 'lmax':lmax})

    
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

        self.fitness, self.acc, self.pneurons = objFun(self.xreal, data, self.params)


    