################## 
# 
# Create a keras model
#   
##################

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

#from keras import backend as k

# Set memory limit
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True 
config.gpu_options.per_process_gpu_memory_fraction = 0.385
session = tf.compat.v1.Session(config=config)


# Returns a model
# Params: MLP params
class MLP:
    # return a keras model
    #Params: l1, l2, eta, input_shape, and num classes from objFun
    @staticmethod
    def build_model(mlp_params):
        l1, l2, eta = mlp_params['l1'], mlp_params['l2'], mlp_params['eta']
        input_shape, c = mlp_params['input_shape'], mlp_params['num_classes']
    
        model = Sequential()
        #print("bye ...")
        #exit()
        # Hidden layers
        model.add(keras.Input(shape=input_shape))
        
        # hidden layer 1
        model.add(Dense(l1, activation='relu'))
        
        # hiden layer 2
        model.add(Dense(l2, activation='relu'))
        
        # Output layer
        model.add(Dense(c, activation='softmax'))
        
        # Build model
        model.build(input_shape)
        
        # Compile model
        # variable with one value: 'sparse_categorical_crossentropy'
        # variable with one-hot encoding: 'categorical_crossentropy'
        #optimizer=SGD(learning_rate=eta, momentum=0.9, nesterov=True)
        model.compile(optimizer=Adam(learning_rate=eta), 
                    loss='sparse_categorical_crossentropy', 
                    metrics=['accuracy'])
        return model
