################## 
# 
# MicroGA Class 
#   
##################

import numpy as np
import copy
import multiprocessing
from threading import Thread
import time
from MLP.mlp import MLP
from obj_fun.obj_fun import get_index_MR
from scipy.spatial import distance

from abstract.abs_metaheuristics import IMetaheuristic
from factory.individuals import IndividualFactory



#class MicroGA(IMetaheuristic):
class MicroGA(IMetaheuristic):
    # params: params: parameters to use in the algorithm
    #         data: a tuple (X,Y) that contains variables or patterns X and the predictors Y.
    def __init__(self, params, data):
        self.params = params
        # train and test
        self.data = data
        del data
        # Population based on individual representation 
        # current pop
        self.POP = None
        # parents pop
        self.PARENTS = None
        # offspring pop
        self.OFFSPRING = None
        # reset pop
        self.AUX_POP = None

        # save the best individual
        self.best_ind = None
        # save fitness, percentage of neurons and accuracy per generation of the best individual
        self.FIT_POP = []
        self.ACC_POP = []
        self.PNEURONS_POP = []

        # number of evaluations
        self.EVALUATIONS = 0
        self.EVALUATIONS_RESET = 0


    # Initialize population
    def init_pop(self):
        npop = self.params['npop']
        # init current pop
        #print("En init_pop")
        self.POP = []
        for i in range(npop):
            self.POP.append(IndividualFactory.build_individual(self.params))
            #print(self.POP[i])
        self.POP = np.array(self.POP)
        #print("Fin init_pop")
        # init parents pop (replaced in selection operator) this only for init
        self.PARENTS = []
        for i in range(npop):
            self.PARENTS.append(IndividualFactory.build_individual(self.params))
        self.PARENTS = np.array(self.PARENTS)

        # init offspring pop
        self.OFFSPRING = []
        for i in range(npop):
            self.OFFSPRING.append(IndividualFactory.build_individual(self.params))
        self.OFFSPRING = np.array(self.OFFSPRING)


    # Initialize aux pop
    def init_aux_pop(self):
        npop = self.params['npop']
        # init current pop
        self.AUX_POP = []
        for i in range(npop):
            self.AUX_POP.append(IndividualFactory.build_individual(self.params))
        self.AUX_POP = np.array(self.AUX_POP)
         

    def print_pop(self, pop:str):
        mes = '\n'+pop+' population'
        print(mes)
        if pop == 'POP':
            for i in range(self.params['npop']):
                #print with __str__
                if self.POP is not None:
                    print(self.POP[i])
                else:
                    print("POP is None", type(self.POP))
        if pop == 'OFFSPRING':  
            for i in range(self.params['npop']):
                #print with __str__
                if self.OFFSPRING is not None:
                    print(self.OFFSPRING[i])
                else:
                    print("OFFSPRING is None", type(self.OFFSPRING))
        if pop == 'PARENTS':    
            for i in range(self.params['npop']):
                #print with __str__
                if self.PARENTS is not None:
                    print(self.PARENTS[i])
                else:
                    print("PARENTS is None", type(self.PARENTS))
            

    # hamming distance
    def meanhamming(self):
        n = self.params['npop']
        nelem = 6  # n * (n - 1) / 2, 6 for four elements
        sum_hamming = 0
        for i in range(n):
            # upper triangular matrix
            for j in range(n - (n - i), n):
                # compute hammming distance between each pair of individuals
                sum_hamming += distance.hamming(self.POP[i].chromosome, self.POP[j].chromosome)
        
        # mean of hamming distance
        mh = sum_hamming / nelem 
    
        return mh


    # Selection: Binary tournament
    def selection(self):
        npop = self.params['npop']
        ind1 = np.random.permutation(npop)

        #print("Permutation:", ind1)
        for i in range(npop-1):
            if self.POP[ind1[i]].fitness > self.POP[ind1[i+1]].fitness:
                self.PARENTS[i] = copy.deepcopy(self.POP[ind1[i]])
            else:
                self.PARENTS[i] = copy.deepcopy(self.POP[ind1[i+1]])
        
        # compare the last one with first
        if self.POP[ind1[-1]].fitness > self.POP[ind1[0]].fitness:
            self.PARENTS[-1] = copy.deepcopy(self.POP[ind1[-1]])
        else:
            self.PARENTS[-1] = copy.deepcopy(self.POP[ind1[0]])



    # Crossover: Two points
    def crossover(self):
        npop = self.params['npop']
        bindim = self.POP[0].params['bindim'] #self.params['bindim']
        print("bindim:", bindim)
        pc = self.params['pc']
        
        # parents to do crossover
        prnt1, prnt2 = 1,3
        
        for _ in range(0, npop, 2):
            r = np.random.random()
            # do crossover
            if r < pc:
                point1, point2 = 0, 0
                while(point1 == point2):
                    point1 = np.random.randint(1, bindim)
                    point2 = np.random.randint(1, bindim)

                # points of crossover
                pc1, pc2 = min(point1,point2), max(point1,point2)
                
                """print("Padres {} y {} se cruzaron".format(prnt1,prnt2))
                print("bindim:", bindim, "  pc1:", pc1, "  pc2:", pc2)

                print("Cromosoma antes de la cruza")
                print(self.PARENTS[prnt1].chromosome)
                print(self.PARENTS[prnt2].chromosome)"""

                aux1 = copy.deepcopy(self.PARENTS[prnt1].chromosome[pc1:pc2+1])
                self.PARENTS[prnt1].chromosome[pc1:pc2+1] = copy.deepcopy(self.PARENTS[prnt2].chromosome[pc1:pc2+1])
                self.PARENTS[prnt2].chromosome[pc1:pc2+1] = aux1

                """print("Cromosoma despues de la cruza")
                print(self.PARENTS[prnt1].chromosome)
                print(self.PARENTS[prnt2].chromosome)"""

            prnt1, prnt2 = 0,2

        self.OFFSPRING = copy.deepcopy(self.PARENTS)
        #print("\nOFFSPRING después de cruza")
        #self.print_pop('OFFSPRING')

    
    # ******** PARALELL EVALUATIONS ********
    # evaluate all individuals in pop
    def evaluate(self, pop:str):
        npop = self.params['npop']
        self.EVALUATIONS += npop

        if pop == 'POP':
            # create and start 4 threads
            threads = []
            for i in range(npop):
                t = Thread(target=self.POP[i].evaluate, args=(self.data,))
                threads.append(t)
            
            # start the threads
            for t in threads:
                t.start()
            
            # wait for the threads to complete
            for t in threads:
                t.join()
            #for i in range(npop):
            #    self.POP[i].evaluate(self.data)

        elif pop == 'OFFSPRING':
            # create and start 4 threads
            threads = []
            for i in range(npop):
                t = Thread(target=self.OFFSPRING[i].evaluate, args=(self.data,))
                threads.append(t)
            # start the threads
            for t in threads:
                t.start()
            # wait for the threads to complete
            for t in threads:
                t.join()
            #for i in range(npop):
            #    self.OFFSPRING[i].evaluate(self.data)
        elif pop == 'AUX_POP':
            self.EVALUATIONS_RESET += 4
            # create and start 4 threads
            threads = []
            for i in range(npop):
                t = Thread(target=self.AUX_POP[i].evaluate, args=(self.data,))
                threads.append(t)
            
            # start the threads
            for t in threads:
                t.start()
            
            # wait for the threads to complete
            for t in threads:
                t.join()
            #for i in range(npop):
            #    self.AUX_POP[i].evaluate(self.data)


    def select_best(self, pop:str):
        npop = self.params['npop']
        idx_best = -1
        max_ = 0

        if pop == 'POP':
            for i in range(npop):
                if self.POP[i].fitness > max_:
                    max_ = self.POP[i].fitness
                    idx_best = i
            # same mem address without copy
            self.best_ind = self.POP[idx_best]
            #print("\nen select_best"+pop)
            #print(self.best_ind)

        elif pop == 'OFFSPRING':
            for i in range(npop):
                if self.OFFSPRING[i].fitness > max_:
                    max_ = self.OFFSPRING[i].fitness
                    idx_best = i
            
            # Update the best solution
            if self.OFFSPRING[idx_best].fitness > self.best_ind.fitness:
                #print("el mejor OFFSPRING fue mejor que el mejor de POP")
                #print(self.OFFSPRING[idx_best])
                # same mem address without copy
                self.best_ind = self.OFFSPRING[idx_best]
            else:
                # Apply elitism
                ind = np.random.randint(0,npop)
                self.OFFSPRING[ind] = self.best_ind
                #print("El mejor de POP fue mejor que el mejor de OFFSPRING")
                #print(self.OFFSPRING[ind])

            #print("\nen select_best"+pop)
            #print(self.best_ind)

        elif pop == 'AUX_POP':
            for i in range(npop):
                if self.AUX_POP[i].fitness > max_:
                    max_ = self.AUX_POP[i].fitness
                    idx_best = i
            
            # Update the best solution
            if self.AUX_POP[idx_best].fitness > self.best_ind.fitness:
                #print("el mejor de AUX_POP fue mejor que el mejor de POP")
                #print(self.AUX_POP[idx_best])
                # same mem address without copy
                self.best_ind = self.AUX_POP[idx_best]
            else:
                # Apply elitism
                ind = np.random.randint(0,npop)
                self.AUX_POP[ind] = self.best_ind
                #print("El mejor de POP fue mejor que el mejor de AUX_POP")
                #print(self.AUX_POP[ind])

            #print("\nen select_best"+pop)
            #print(self.best_ind)
        

        #return idx_best

    # save fitness, accuracy and pneurons of the best_ind in tmp archive
    def add_history_results(self):
        self.FIT_POP.append(self.best_ind.fitness)
        self.ACC_POP.append(self.best_ind.acc)
        self.PNEURONS_POP.append(self.best_ind.pneurons)

        try:
            if not "num_of_FS" in self.params:
                num_FS = []
                num_FS.append(self.best_ind.num_of_FS)
                self.params.update({"num_of_FS" : num_FS})
            else:
                self.params["num_of_FS"].append(self.best_ind.num_of_FS)
        except AttributeError:
            pass

    # Evaluate best individual neural network
    def evaluate_best(self, test):
        print("Evaluate the best individual on test (unseen) data")
        # train
        Xtr, Ytr = self.data[0], self.data[1]
        # test
        Xtt, Ytt = test[0], test[1]

        #params
        l1 = self.best_ind.xreal[0]
        l2 = self.best_ind.xreal[1]
        # learning rate
        # if None, eta is optimized 
        if self.params['eta'] == None:
            eta = self.best_ind.xreal[2]
        else:
            eta = self.params['eta']

        # if method include feature selection
        method = self.params['method_name']
        method_FS = method.split('-')[-1]
        if method_FS == "FS" or method_FS == "MRMR":
            # variables to optimize, in addition to FS
            opt = self.params['optimize']
            # Features selected x[3:]
            index = self.best_ind.xreal[opt:]
            FS = np.where(index == 1)
            FS = FS[0]
            Xtr = Xtr[:,FS]
            Xtt = Xtt[:,FS]

        # if is a feature selection based on relevance (MR)
        if "MR" in method:
            index_MR = get_index_MR(None, None)
            FS = int(self.best_ind.xreal[3]) # num of features selected
            print("numero de features:", FS)
            FS = index_MR[:FS] # get first FS'th selected indexes
            #print("\nfeatues seleccionadas\n", FS)
            Xtr = Xtr[:,FS]
            Xtt = Xtt[:,FS]
        
        # input shape
        input_shape = np.shape(Xtr[0])
        print("input shape :", input_shape)
        epochs = self.params['epochs']
        batch_size = self.params['batch_size']

        mlp_params = {'l1':l1, 'l2':l2, 'eta':eta, 'input_shape':input_shape, 'num_classes':self.params['num_classes']}
        model = MLP.build_model(mlp_params)
        model.summary()
        print("After summary")

        history_train = model.fit(Xtr,Ytr, epochs=epochs, verbose=1, batch_size=batch_size, shuffle=True)
        self.best_ind.history_train = history_train
        
        self.best_ind.eval_test = model.evaluate(Xtt, Ytt)
        print("end of evaluate_best")


    # return best individual (bbest, xbest, fbest), model.h5 and history train, y_pred, f1_score, precision, and recall
    #        fitness_curves, acc_curves, pneurons_curves
    def run(self):
        gen = self.params['gen']
        npop = self.params['npop']
        threshold_limit = self.params['threshold']
        
        #self.print_pop(pop)
        # Init population
        self.init_pop()
        

        #print("\n"+pop+" despues de init_pop")
        #self.print_pop(pop)
        #pool_ = multiprocessing.Pool(processes=4)
        #inputs = ['POP' for i in range(npop)]
        #pool_.map(self.evaluate, inputs)

        # evaluate current pop
        self.evaluate('POP')
        print("Despues de evaluarse POP")
        self.print_pop('POP')

        #print("bye ...")
        #exit()
        

        # Current best solution
        self.select_best('POP')
        #print("\nEL mejor individuo de POP")
        #print(self.best_ind)

        self.add_history_results()

        #print("El de POP")
        #print(self.POP[idx_best])
        #print("el de POP (again)")
        #print(self.POP[idx_best])
        
        for g in range(gen-1):
            print('gen ', g)

            #print("\nPARENTS antes de selection")
            #self.print_pop('PARENTS')
            # Selection: Binary tournament
            self.selection()

            #print("\n despues de selection")
            #self.print_pop('PARENTS')

            # Crossover: Two points
            self.crossover()

            # evaluate offspring
            self.evaluate('OFFSPRING')
            print("OFFSPRING despues de evaluate")
            self.print_pop('OFFSPRING')
            # Best solution of the offspring
            self.select_best('OFFSPRING')
            

            # Next generation (replace)
            self.POP = copy.deepcopy(self.OFFSPRING)

            #print("\nLa nueva poblacion POP = OFFSPRING")
            #self.print_pop('POP')

            #print("\n check convergence")
            # check convergence
            mh = self.meanhamming()

            if mh < threshold_limit:
                print("convergence! gen", g)
                print("mh:", mh)

                # reset population
                self.init_aux_pop()

                # evaluate reset pop
                self.evaluate('AUX_POP')

                # Best solution of the reset population
                self.select_best('AUX_POP')

                # Next generation (replace)
                self.POP = copy.deepcopy(self.AUX_POP)
                
                print("POP = AUX_POP")
                self.print_pop('POP')
                #print("\nbye..")
                #exit()
            else:
                print("Not convergence")

            
            # add fitness, acc and pneurons to history array
            self.add_history_results()


        return self.FIT_POP, self.ACC_POP, self.PNEURONS_POP





