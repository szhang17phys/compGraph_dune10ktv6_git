#Simplified by Shu, 20250430
#Only for FD1 2000 opchs---


import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input, Model
from tensorflow.keras.utils  import plot_model
from tensorflow.keras.layers import Dense, concatenate, Multiply, Lambda, Flatten, BatchNormalization, PReLU, ReLU
from tensorflow.keras.optimizers import SGD, Adam

import numpy as np

def outer_product(inputs):
    x, y = inputs
    batchSize = K.shape(x)[0]    
    outerProduct = x[:,:, np.newaxis] * y[:, np.newaxis,:]
    return outerProduct
    


#Suggested by ChatGPT, to fix underestimation of high y_true region---
#Soft Weighting Based on y_true
def vkld_loss(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.0)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0)

    kl_term = (y_true - y_pred) * K.log(y_pred / y_true)

    # Smooth weight: 1.0 + log(y_true + 1e-3), clipped to 1.0 min
    log_weight = K.log(y_true + 1e-3)
    boost_weight = 1.0 + K.relu(log_weight)  # ensure weights ≥ 1.0

    weighted_kl = kl_term * boost_weight
    return K.abs(K.sum(weighted_kl, axis=-1))




#Network Architecture
#Input: three scalars (Position of scintillation)
#Output: one vecotr (photon detector response)
#===================================================================
#For FD1 (dune10kt_v6) geometry, containing 6000 optical channels---
#Due to symmetry, here consider 2000 opchs---
#Written by Shu, 20241028---
def model_fdhdv6_dune10kt(dim_pdr):
    pos_x = Input(shape=(1,), name='pos_x')  # explicitly intensity [364,726]
    pos_y = Input(shape=(1,), name='pos_y')  # explicitly vertical [-600,600]
    pos_z = Input(shape=(1,), name='pos_z')  # explicitly horizontal [0,5810]
    input_layer = [pos_x, pos_y, pos_z]
    
    # Explicit input normalization clearly stated:
    x_scaled = Lambda(lambda x: (x - 364) / 362, name='x_scaled')(pos_x)
    y_scaled = Lambda(lambda y: (y + 600) / 1200, name='y_scaled')(pos_y)
    z_scaled = Lambda(lambda z: z / 5810, name='z_scaled')(pos_z)

    feat_int = Dense(1)(x_scaled)#Shu: It can only be 1, due to L1049---
    feat_int = BatchNormalization(momentum=0.9)(feat_int)
    feat_int = ReLU()(feat_int)
        
    feat_row = Dense(10)(y_scaled)
    feat_row = BatchNormalization(momentum=0.9)(feat_row)
    feat_row = ReLU()(feat_row)
    
    feat_col = Dense(12)(z_scaled)
    feat_col = BatchNormalization(momentum=0.9)(feat_col)
    feat_col = ReLU()(feat_col)
    
    feat_cov = Lambda(outer_product)([feat_row, feat_col])
    feat_cov = Flatten()(feat_cov)
    feat_cov = Multiply()([feat_cov, feat_int])
    
    feat_cov = Dense(640)(feat_cov)
    feat_cov = BatchNormalization(momentum=0.9)(feat_cov)
    feat_cov = ReLU()(feat_cov)
    
    pdr      = Dense(dim_pdr, activation='sigmoid', name='vis_full')(feat_cov)
    model    = Model(inputs=input_layer, outputs=pdr, name='fdhdv6_dune10kt_model')
    
    model.summary()
    return model

