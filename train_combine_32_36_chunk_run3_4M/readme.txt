#Created by Shu, 20250430

1. Based on train 31

2. Combination of train 32 and 36:

3. (train 32) Normalize input: 
    pos_x = Input(shape=(1,), name='pos_x')  # explicitly intensity [364,726]
    pos_y = Input(shape=(1,), name='pos_y')  # explicitly vertical [-600,600]
    pos_z = Input(shape=(1,), name='pos_z')  # explicitly horizontal [0,5810]
    input_layer = [pos_x, pos_y, pos_z]
    
    # Explicit input normalization clearly stated:
    x_scaled = Lambda(lambda x: (x - 364) / 362, name='x_scaled')(pos_x)
    y_scaled = Lambda(lambda y: (y + 600) / 1200, name='y_scaled')(pos_y)
    z_scaled = Lambda(lambda z: z / 5810, name='z_scaled')(pos_z)

4. (train 36) #Suggested by ChatGPT, to fix underestimation of high y_true region---
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

5. chunk method (to deal with memory over-consumption); utlis.py is modified---

    
