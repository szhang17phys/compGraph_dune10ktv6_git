# utlities
import os
import re
import sys
import time

import tensorflow as tf

import pickle as pk
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
from tensorflow.python.framework.graph_util import convert_variables_to_constants

from pylab import *
from networks import *



def mkdir(dir="image/"):
       if not os.path.exists(dir):
           print('make directory '+str(dir))
           os.makedirs(dir)


    


def get_data(path, nfile, dim_pos, dim_pdr):
    dataset = []
    files = [f for f in os.listdir(path)]
    print('Processing ' + str(len(files) if nfile == -1 or nfile > len(files) else nfile) + ' files...')
    for i,f in enumerate(files):
        if i == nfile:
            break
        datafile = os.path.join(path, f)
        datatmp  = []
        with open(datafile, 'rb') as ft:
            datatmp = pk.load(ft)
            dataset.extend(datatmp)
    n_vec = len(dataset)
    print('Dataset loaded, dataset length: '+str(n_vec))
    
    inputs  = np.zeros(shape=(n_vec, dim_pos))
    outputs = np.zeros(shape=(n_vec, dim_pdr))
    for i in range(0, n_vec):
        event = dataset[i] 
        inputs[i,0] = event['x']
        inputs[i,1] = event['y']
        inputs[i,2] = event['z']
        outputs[i]  = event['image'].reshape(dim_pdr)
    return inputs, outputs






   



#Suggested by ChatGPT, 20241228---
def lr_scheduler(epoch, lr):
    #Warm-up phase for the first 10 epochs
    if epoch < 10:
        lr = 1e-5 + (2e-4 - 1e-5) * (epoch / 10)  #Linearly ramp up from 1e-5 to 2e-4

    # After 1000 epochs, decay the learning rate by a factor of 0.997 for each epoch
    elif epoch < 1000:
        lr = 2e-4  # Set learning rate to 2e-4 for the first 1000 epochs
    else:
        lr = lr * 0.997  #Apply decay

    return lr



def train(pos, pdr, mtier, epochs, batchsize, modpath, opt):
    dim_pdr = pdr.shape[1]
    
    #load different models according to the number of optical channels
    if dim_pdr == 2000:
        if mtier == 0:
            print('Loading fdhdv6v6_dune10kt 2000 opch net...')
            model = model_fdhdv6_dune10kt(dim_pdr) 


    if opt == 'SGD':
        optimizer = SGD(momentum=0.9)
    else:
        optimizer = Adam()
        
    model.compile(optimizer=optimizer, loss=vkld_loss, metrics=['mape', 'mae'])


    #if there is existing weight file, load it.
    weight = modpath+'best_model.h5'
    if os.path.isfile(weight):
        model.load_weights(weight)    
        

    #Suggested by ChatGPT, 20241228---
    checkpoints = [
        #Save the best model based on validation loss with save_freq as 'epoch' and mode='min'
        ModelCheckpoint(weight, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_freq='epoch'),
        #Learning rate scheduler with warm-up and decay (as modified earlier)
        LearningRateScheduler(lr_scheduler),
        #Early stopping with a reduced patience value to avoid long training if no improvement
        EarlyStopping(monitor='val_loss', patience=150, restore_best_weights=True)
    ]
                 

    #start model training
    ftrain, ftest, ptrain, ptest=train_test_split(pos, pdr, test_size=0.15)
    model.fit({'pos_x': ftrain[:,0], 'pos_y': ftrain[:,1], 'pos_z': ftrain[:,2]}, ptrain,
               validation_data=({'pos_x': ftest[:,0], 'pos_y': ftest[:,1], 'pos_z': ftest[:,2]}, ptest),
               epochs=epochs, batch_size=batchsize, callbacks=checkpoints, verbose=2, shuffle=True)
               

    #export trained model in SavedModel format for C++ API
    tf.saved_model.save(model, modpath)
    

    #initial evaluation of the training performance
    model.load_weights(weight)
    ypred = model.predict({'pos_x': ftest[:,0], 'pos_y': ftest[:,1], 'pos_z': ftest[:,2]})
    n_test = len(ftest)
        
    



