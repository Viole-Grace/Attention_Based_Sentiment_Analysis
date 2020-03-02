import numpy as np
import pandas as pd

import keras.backend as K
from keras.layers import Layer
from keras.models import Sequential, Model

class Attention(Layer):

    def __init__(self, **kwargs):

        super(Attention,self).__init__(**kwargs)
    
    def build(self, input_shape):

        self.W = self.add_weight(name='att_wt', shape=(input_shape[-1],1), initializer='normal')
        self.b = self.add_weight(name='att_b', shape=(input_shape[1],1), initializer='zeros')
        super(Attention, self).build(input_shape)

    def call(self,x):

        mlp = K.squeeze(K.tanh(K.dot(x, self.W)+self.b), axis=-1)
        at = K.expand_dims(K.softmax(mlp), axis=-1)
        output = x*at
        return K.sum(output, axis=1) #context vector
    
    def compute_output_shape(self, input_shape):

        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention,self).get_config()
