from __future__ import division
import tensorflow as tf
# from keras.models import Model, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Activation, Lambda, Add, Dense
import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')

def ISFE_CA(x, feature_size, kernel_size):
    x = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    x = Activation('relu')(x) 
    
    return x




def ISFE(x, feature_size, kernel_size, scale_ISFE=0.1):
    x = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    
    x = Activation('relu')(x) 
    
    # return scale_ISFE*x
    
    return Lambda(lambda x: x * scale_ISFE)(x)


def resBlock(x, feature_size, kernel_size, scale_Res=0.1):
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale_Res)(tmp)
    return Add()([x, tmp])



def PSRBlock(x, feature_size, kernel_size, scale_ISFE = 0.1, scale_Res = 0.1, Num_resBlocks = 3):
    x = ISFE(x, feature_size, kernel_size, scale_ISFE=0.1)
    for i in range(Num_resBlocks):
        x = resBlock(x, feature_size, kernel_size, scale_Res=0.1)   
    
    # x = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(x)
    
    #波段加权
    # x = ISFE_CA(x, 6, kernel_size, scale_ISFE=0.1)
    
    
    
    return x

def SFFBlock(feature10, feature20, feature60, kernel_size, feature_size):
    x = Concatenate(axis=3)([feature10, feature20])
    if feature60:
        x = Concatenate(axis=3)([feature10, feature20, feature60])
    
    
    x = Conv2D(feature_size, kernel_size, kernel_initializer='he_uniform', padding='same')(x) 
      
    fc = Dense(feature_size, activation = 'relu')
    x = fc(x)
    x = fc(x)
     
    # x = Conv2D(feature_size, kernel_size=[3, 3], kernel_initializer='he_uniform', padding='same')(x)   
    return x


def RCA_Block( x, kernel_size, channel):
    
    _res = x
    
    x = Conv2D(channel, kernel_size,  padding='same', activation=tf.nn.relu)(x)
    x = Conv2D(channel, kernel_size, padding='same')(x)


    x = tf.add(_res, x)
       
    return x


def Residual_Group( x, kernel_size, n_RCAB, channel):
    skip_connection = x
    
    for i in range(n_RCAB):
        x = RCA_Block( x, kernel_size, channel)
    
    x = Conv2D(channel, kernel_size, padding='same')(x) 
    x = x + skip_connection   
    return x



def SPRmodel(input_shape, kernel_size, scale_ISFE, scale_Res, Num_resBlocks, feature_size=64):

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
 
        
    feature10 = PSRBlock(input10, feature_size, kernel_size, scale_ISFE = 0.1, scale_Res = 0.1, Num_resBlocks = 3)
    feature20 = PSRBlock(input20, feature_size, kernel_size, scale_ISFE = 0.1, scale_Res = 0.1, Num_resBlocks = 3)
    if len(input_shape) == 3:
        feature60 = PSRBlock(input60, feature_size, kernel_size, scale_ISFE = 0.1, scale_Res = 0.1, Num_resBlocks = 3)
    elif len(input_shape) == 2:
        feature60 = None
    
    
    x = SFFBlock(feature10, feature20, feature60, kernel_size, feature_size)
    x = Conv2D(6, kernel_size, kernel_initializer='he_uniform', padding='same')(x) 
    
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
              
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model
