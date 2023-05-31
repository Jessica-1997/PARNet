from __future__ import division
# from keras.models import Model, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Lambda, Add
import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')


    
            
def SRCNN_model(input_shape, out_channels):
    
    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=3)([input10, input20, input60])
    else:
        x = Concatenate(axis=3)([input10, input20])
   
    x = Conv2D(64, (9, 9), strides=(1, 1), activation='relu',padding='same')(x)
    x = Conv2D(32, (1, 1), strides=(1, 1), activation='relu',padding='same')(x)
    x = Conv2D(out_channels, (5, 5), strides=(1, 1), padding='same')(x)

    if len(input_shape) == 3:
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:  
        model = Model(inputs=[input10, input20], outputs=x)
    return model









