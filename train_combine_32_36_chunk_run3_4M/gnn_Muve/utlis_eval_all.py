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

from mpl_toolkits.mplot3d import Axes3D # draw 3d figure---

from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit

#Suggested by ChatGPT, 20250121---
#from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from pylab import *
from networks import *





#========================================
#Important variables---
pos_X = []
pos_Y = []
pos_Z = []
value_bias = [] #used to store the biased values---
value_true = [] #used to store simu (geant4) values---
value_emul = [] #used to store emul (graphModule) values---






#========================================
#Make directory---
def mkdir(dir="image/"):
       if not os.path.exists(dir):
           print('make directory '+str(dir))
           os.makedirs(dir)






#============================================
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








#====================================================
#For dune10kt_v6, 2000 opchs---
#Written by Shu, 20241101---
#Based on previous protodune_vd code---
def eval_fdhdv6_dune10kt(pos, pdr, pre, evlpath):                
    print('\n\n\nSingle Emission Vertex Evaluation========================')

    #scan along z direction---            
    cut_x  = (pos[:,0] > 501) & (pos[:,0] < 502)#Ex: [501, 502], [684, 685],  [397, 398], [509, 510] 
    cut_y  = (pos[:,1] > 304) & (pos[:,1] < 305)  #Ex: [304, 305], [-39, -38],  [-5, -4], [290, 291]
    cut_z  = (pos[:,2] > 10) & (pos[:,2] < 2000) #Ex: [10, 2000], [900, 1000], [0, 3100], [1950, 2050]

    true_pds = pdr[cut_x & cut_y & cut_z]#GEANT4---
    emul_pds = pre[cut_x & cut_y & cut_z]#model---


    print("Values of CutX: ", pos[:,0][cut_x & cut_y & cut_z])
    print("Length of CutX: ", len(pos[:,0][cut_x & cut_y & cut_z]))

    print("\nValues of CutY: ", pos[:,1][cut_x & cut_y & cut_z])
    print("Length of CutY: ", len(pos[:,1][cut_x & cut_y & cut_z]))

    print("\nValues of CutZ: ", pos[:,2][cut_x & cut_y & cut_z])
    print("Length of CutZ: ", len(pos[:,2][cut_x & cut_y & cut_z]))

    print("\nSize of GEANT4: ", len(true_pds))
    print(true_pds)
    np.savetxt("singleVertex_GEANT4.txt", true_pds.flatten(), fmt="%.6f")

    print("Size of MODULE: ", len(emul_pds))
    print(emul_pds)
    np.savetxt("singleVertex_Module.txt", emul_pds.flatten(), fmt="%.6f")
    print("=============================================================")

#====================================================











        
def eval(pos, pdr, mtier, modpath, evlpath):
    dim_pdr = pdr.shape[1]

    #Added by Shu, for FDHD_v4 2000 opch, 20241101---
    if dim_pdr == 2000:
        if mtier == 0:
            print('\nLoading fdhdv6_dune10kt 2000 opch net (MODULE)...\n')
            model = model_fdhdv6_dune10kt(dim_pdr)


 
    weight = modpath+'best_model.h5'
    if os.path.isfile(weight):
        print('\nLoading weights...\n')
        model.load_weights(weight)
    else:
        print('Err: no weight found!')
        return
        
    print('Predicting...')
    tstart = time.time()

    #The prediction of model------
    pre = model.predict({'pos_x': pos[:,0], 'pos_y': pos[:,1], 'pos_z': pos[:,2]})
    print('\n')
    print( '\nFinish evaluation in '+str(time.time()-tstart)+'s.')
    

    #Added by Shu, for FDHD_v6 2000 opch, 20241101---
    if dim_pdr == 2000:
        eval_fdhdv6_dune10kt(pos, pdr, pre, evlpath)

#====================================================================













#Shu: This part (expecially region of x_list) should fit the exact geometry!!!---
#Shu: Nov 2, 2024---
#===Case of sum up all XArapucas=======================================
    print('\n--------------------Scan along X-------------------------')
    print('Intensity and resolution evaluating...')            
    pre = pre.sum(axis=1)#it means suming up all pds; YES, refer to Mu's paper---
    pdr = pdr.sum(axis=1)
    
    cut = (pre != 0) & (pdr != 0)

    #Shu: Important------------------------------
    x_list = [400, 600, 800]
    #--------------------------------------------

    for i in range(len(x_list)):
        w = True
        if i == 0:#here range of random abs(x) is [0. 100]---
            low_x = np.absolute(pos[:,0]) >  0#the x position of the vertex---
            upp_x = np.absolute(pos[:,0]) <= x_list[i]
            title = '0<|x|<%d' %(x_list[i])
        elif i == (len(x_list)-1):#here range of random x is [0, x_list[i-1]]---
            low_x = np.absolute(pos[:,0]) >  0
            upp_x = np.absolute(pos[:,0]) <= x_list[i]
            #title = 'All (%d<|x|<%d)' %(0, x_list[i])
            title = 'Vertex of Whole Space'
            w     = False
        else:#here range of random x is [x_list[i-1], x_list[i]]---
            low_x = np.absolute(pos[:,0]) <= x_list[i]
            upp_x = np.absolute(pos[:,0]) >  x_list[i-1]
            title = '%d<|x|<%d' %(x_list[i-1], x_list[i])
            
        image_diff = pre[cut & low_x & upp_x] - pdr[cut & low_x & upp_x]
        image_true = pdr[cut & low_x & upp_x]
        image_emul = pre[cut & low_x & upp_x]
        visib_diff = np.divide(image_diff, image_true)
        

#        savehist(visib_diff, (-1, 1), '(Emul-Simu)/Simu', 'Counts', title, evlpath+'X_intensity-'+str(x_list[i]), w) 
        
        print("\nBig biased values:")
        Bias = 0
        for diff in visib_diff:
            if abs(diff) > 1:
                print(str(format(diff, '.3f')))
                Bias += 1
        print("# of abs((emul-simu)/simu)>1 : ", Bias)

    print('---------------------------------------------------------')
#        print("Length of image_diff: ", len(image_diff))
#        print("iamge_diff: \n",image_diff)
#---------------------------------------------------------------------

#=====================================================================







#======CORE=================================

#Extract data and keep them into txt file=============================
#===Test vertex of large (emul-simu)/simu=============================
    print("\n\n\n")
    print("===Deal with biased values==================")
    print("Total points (Length of visib_diff): ", len(visib_diff))
    print("visib_diff: ", visib_diff)

    for nums in range(0, len(visib_diff)):
        pos_X.append(pos[nums, 0])
        pos_Y.append(pos[nums, 1])
        pos_Z.append(pos[nums, 2])
        value_bias.append(visib_diff[nums])
        value_true.append(image_true[nums]*1000000)
        value_emul.append(image_emul[nums]*1000000)

            #print(visib_diff[nums])


    #y and x---
    plt.scatter(pos_Y, pos_X, c='red', s=1)
    plt.xlabel('Y')
    plt.ylabel('X')
    plt.grid()
    plt.savefig('dis_Y_X.png', dpi=200)
    #y and z---
    plt.scatter(pos_Y, pos_Z, c='red', s=1)
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.grid()
    plt.savefig('dis_Y_Z.png', dpi=200)
    #x and z---
    plt.scatter(pos_X, pos_Z, c='red', s=1)
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.grid()
    plt.savefig('dis_X_Z.png', dpi=200)
    #x, y and z---
    figxyz = plt.figure()
    ax = Axes3D(figxyz)
    ax.scatter(pos_X, pos_Y, pos_Z)    
    plt.savefig('dis_X_Y_Z.png', dpi=200)

   #keep x, y, z & values in  txt files---
    with open('all_xPos.txt', 'w') as filehandle:
        for listitem in pos_X:
            filehandle.write('%f\n' % listitem)

    with open('all_yPos.txt', 'w') as filehandle:
        for listitem in pos_Y:
            filehandle.write('%f\n' % listitem)

    with open('all_zPos.txt', 'w') as filehandle:
        for listitem in pos_Z:
            filehandle.write('%f\n' % listitem)

    with open('all_biasValues.txt', 'w') as filehandle:
        for listitem in value_bias:
            filehandle.write('%f\n' % listitem)

    with open('all_simuValues.txt', 'w') as filehandle:
        for listitem in value_true:
            filehandle.write('%d\n' % listitem)

    with open('all_emulValues.txt', 'w') as filehandle:
        for listitem in value_emul:
            filehandle.write('%d\n' % listitem)

    print("Length of pos_X: ", len(pos_X))
    print("pos[:, 0] ", pos[:, 0])
    print("pos[:, 1] ", pos[:, 1])
    print("pos[:, 2] ", pos[:, 2])
    print("---------------------------------------------------")

#=================================================================





