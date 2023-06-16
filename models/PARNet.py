import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, Lambda, Add
from tensorflow.keras.activations import sigmoid
import tensorflow.keras.backend as K
K.set_image_data_format('channels_first')

    
def PA(nf, x):
    '''PA is pixel attention'''
    y = Conv2D(nf, kernel_size =1, strides=(1, 1), padding='same')(x)
    y = sigmoid(y)
    out = tf.multiply(x, y)
    return out

        
def SCPA(x, nf, reduction):
   
    residual = x
    out_b = Conv2D(nf, kernel_size=3, strides=(1, 1), padding='same')(x)
    out_b = tf.nn.leaky_relu(out_b, alpha=0.2)
    out_b = Lambda(lambda x: x * 0.1)(out_b)
    
    out_a = Conv2D(nf, kernel_size =3, strides=(1, 1), padding='same')(x)
    out_a = tf.nn.leaky_relu(out_a, alpha=0.2)
    out_a = Lambda(lambda x: x * 0.1)(out_a)
    
    out = Concatenate(axis=3)([out_a, out_b])
    out = Conv2D(nf, kernel_size=1, strides=(1, 1), padding='same')(out)

    out =out + residual
    return out

def PAN_model(input_shape, out_channels, feature_size, nblocks): 

    input10 = Input(shape=input_shape[0])
    input20 = Input(shape=input_shape[1])
    if len(input_shape) == 3:
        input60 = Input(shape=input_shape[2])
        x = Concatenate(axis=3)([input10, input20, input60])
    else:
        x = Concatenate(axis=3)([input10, input20])
         

    out = Conv2D(feature_size, 3, strides=(1, 1), padding='same')(x)
      
    fea = out

    for i in range(nblocks):
        out = SCPA(out, feature_size, reduction = 2)
        fea = Concatenate(axis=3)([fea, out])
    out = Conv2D(feature_size, 3, strides=(1, 1), padding='same')(fea)    

    out = PA(feature_size, out)  
    out = tf.nn.leaky_relu(out, alpha=0.2)  
        
   
    out = Conv2D(feature_size, 3, strides=(1, 1), padding='same')(out)
    out = tf.nn.leaky_relu(out, alpha=0.2)
    
    out = Conv2D(feature_size, 3, strides=(1, 1), padding='same')(out)
    out = tf.nn.leaky_relu(out, alpha=0.2)
 
    out = Conv2D(out_channels, 3, strides=(1, 1), padding='same')(out)   
    out = Lambda(lambda x: x * 0.2)(out)

    if len(input_shape) == 3:
        out = Add()([out, input60])
        model = Model(inputs=[input10, input20, input60], outputs=out)
    else:               
        out = Add()([out, input20])
        model = Model(inputs=[input10, input20], outputs=out)

    return model


