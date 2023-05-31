from __future__ import division
# from keras.models import Model, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Activation, Lambda, Add
import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')


def resBlock(x, channels, kernel_size=[3, 3], scale=0.1):
    tmp = Conv2D(channels, kernel_size,
                 kernel_initializer='he_uniform', padding='same')(x)
    tmp = Activation('relu')(tmp)
    tmp = Conv2D(channels, kernel_size,
                 kernel_initializer='he_uniform', padding='same')(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def s2model(input_shape, out_channels, num_layers=32, feature_size=256):

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=3)([input10, input20, input60])
    else:
        # TensorShape([None, 10, 32, 32])
        x = Concatenate(axis=3)([input10, input20])

    # Treat the concatenation
    x = Conv2D(feature_size, (3, 3), kernel_initializer='he_uniform',
               activation='relu', padding='same')(x)
    #TensorShape([None, 128, 32, 32])
    for i in range(num_layers):
        x = resBlock(x, feature_size)  # TensorShape([None, 128, 32, 32])
    
    x = Conv2D(out_channels, (3, 3), kernel_initializer='he_uniform', padding='same')(x)
    # x = Dropout(0.3)(x)
    if len(input_shape) == 3:
        x = Add()([x, input60])
        model = Model(inputs=[input10, input20, input60], outputs=x)
    else:
              
        x = Add()([x, input20])
        model = Model(inputs=[input10, input20], outputs=x)
    return model
