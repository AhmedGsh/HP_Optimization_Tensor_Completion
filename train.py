import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import losses

import time 
import random
import numpy as np

'''
Specify the combination of number of output classes, 
and the configuration of hyperparameters.

learn_rate:[]
neur:[]
momentum:[0,1]
num_hidden_layers:[1,5]
batch_size:[]
num_epoch:[]
'''
def train_model(x_tr,y_tr,num_class,learn_rate,neur,momentum,num_hidden_layers=1,batch_size=32,num_epoch=2): 
    
    train_start = time.time()    
    
    model = Sequential()
    model.add(Flatten())
    for i in range(num_hidden_layers):
        model.add(Dense(neur, activation='relu'))
    model.add(Dense(num_class, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=learn_rate, momentum=momentum),
                  metrics=['accuracy'])


    # train_perf_prior_train = model.evaluate(x_tr, y_tr, verbose=0)
    #val_perf_prior_train = model.evaluate(Xs_val, Ys_val, verbose=0)

    
    history = model.fit(x_tr, y_tr,
              batch_size=batch_size,
              epochs=num_epoch,
              verbose=1#,
              #validation_data=(Xs_val, Ys_val)
                )
    train_stop = time.time()
    train_time = train_stop - train_start
    
    return model, train_time