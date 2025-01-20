# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 15:01:52 2020

@author: Stacia16
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K, optimizers
from tensorflow.keras import initializers, regularizers, constraints
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Activation, Conv1D, Conv2D, UpSampling2D
from tensorflow.keras.layers import  Multiply, Add, Layer, InputSpec, ZeroPadding1D, ZeroPadding2D, BatchNormalization, AveragePooling1D, AveragePooling2D
from tensorflow.python.keras.utils import conv_utils #Change import path to reflect updated location in Tensorflow
from tensorflow.python.ops import control_flow_ops
from operator import sub

class InstanceNormalization(Layer):
    """Instance normalization layer.
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each
            instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a Sequential model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](
        https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
 
    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')
 
        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')
 
        self.input_spec = InputSpec(ndim=ndim)
 
        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)
 
        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True
 
    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))
 
        if self.axis is not None:
            del reduction_axes[self.axis]
 
        del reduction_axes[0]
 
        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev
 
        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]
 
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed
 
    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
 
class Adaptive_AvgPooling2D:
    
    def __init__(self, output_size):
        
        self.output_h, self.output_w = output_size
        
    def __call__(self, inputs):
        batchsize, height, width, channel = K.int_shape(inputs)
        
        stride_step = np.floor(np.array([height/self.output_h, width/self.output_w]))
        kernel_size = np.array([height,width]) - (np.array([self.output_h, self.output_w])-1)*stride_step
        
        Adp_out = AveragePooling2D(pool_size = tuple(kernel_size.astype('int')), strides = tuple(stride_step.astype('int')))(inputs)
               
        return Adp_out

class Adaptive_AvgPooling1D:
    
    def __init__(self, output_size):
        
        self.output_size = output_size
        
    def __call__(self, inputs):
        batchsize, sequence, features = K.int_shape(inputs)
        stride_step = np.floor(sequence/self.output_size)
        kernel_size = sequence-(self.output_size-1)*stride_step
        Adp_out = AveragePooling1D(pool_size = tuple(kernel_size.astype('int')), strides = tuple(stride_step.astype('int')))(inputs)
        
        return Adp_out

    
class Adaptive_Upsampling2D:
    
    def __init__(self, size=(2, 2), data_format='channels_last', interpolation='bilinear', **kwargs):
    
      self.size = size
      self.data_format = data_format
      if interpolation not in {'nearest', 'bilinear'}:
        raise ValueError('`interpolation` argument should be one of `"nearest"` '
                          'or `"bilinear"`.')
      self.interpolation = interpolation
      self.input_spec = InputSpec(ndim=4)
    
    def __call__(self, inputs):
        
        return tf.image.resize(inputs,[self.size[0], self.size[1]], method=self.interpolation)

    
class GatedConv1D:

    def __init__(self, output_dim, kernel_size, kwargs_conv={}, kwargs_gate={}):
        self.conv = Conv1D(output_dim, kernel_size, **kwargs_conv) 
        self.conv_gate = Conv1D(output_dim, kernel_size, activation="sigmoid", **kwargs_gate)
        self.conv_resize = Conv1D(output_dim, kernel_size = 1, strides=1)
        self.pad_input = ZeroPadding1D(padding=(kernel_size-1, 0))
    
    def __call__(self, inputs, res_input = None):
        X = self.pad_input(inputs)
        A = self.conv(X)
        A = InstanceNormalization()(A)
        if res_input is not None:
            if K.int_shape(A)[-1] != K.int_shape(res_input)[-1]:
                res_input =self.conv_resize(res_input)
            A = Add()([res_input,A])
            
        B = self.conv_gate(X)
        B = InstanceNormalization()(B)
        return Multiply()([A, B])
  
    
class GatedConv2D(Layer):

    def __init__(self, output_dim, kernel_size, kwargs_conv={}, kwargs_gate={}):
        self.conv = Conv2D(output_dim, kernel_size, **kwargs_conv) 
        self.conv_gate = Conv2D(output_dim, kernel_size, activation="sigmoid", **kwargs_gate)
        self.conv_resize = Conv2D(output_dim, kernel_size =(1,1), strides=(1,1))
        self.pad_input = ZeroPadding2D(padding=(tuple(map(sub, kernel_size, (2,2)))))
    
    def __call__(self, inputs, res_input = None):
        X = self.pad_input(inputs)
        A = self.conv(X)
        A = InstanceNormalization()(A)
        if res_input is not None:
            if K.int_shape(A)[-1] != K.int_shape(res_input)[-1]:
                res_input =self.conv_resize(res_input)
            A = Add()([res_input,A])
            
        B = self.conv_gate(X)
        B = InstanceNormalization()(B)
        return Multiply()([A, B])
    
    
class Attention_gate_v1:

    def __init__(self, F_int, kwargs_conv={}, kwargs_gate={}):
          
        self.conv_W_xl = Conv2D(F_int, kernel_size=1,strides=1, padding='valid', **kwargs_conv) #theta
        self.conv_W_xg = Conv2D(F_int, kernel_size=1,strides=1, padding='valid', **kwargs_conv) #phi
        self.conv_W_psi = Conv2D(1, kernel_size=1,strides=1, padding='valid', **kwargs_conv)

    def __call__(self, xl, xg):
        
        xl_bs, xl_h, xl_w, xl_ch = tf.shape(xl)
        
        theta = self.conv_W_xl(xl)
        # theta = BatchNormalization()(theta)
        theta_bs, theta_h, theta_w, theta_ch = tf.shape(theta)

        phi = self.conv_W_xg(xg)
        phi_bs, phi_h, phi_w, phi_ch = tf.shape(phi)
        phi = Adaptive_Upsampling2D(size=(theta_h, theta_w))(phi)
        # W_xg = BatchNormalization()(W_xg)

        psi = Add()([theta,phi])
        psi = Activation('relu')(psi)
        psi = self.conv_W_psi(psi)
        psi = Activation('sigmoid')(psi)
        

        psi_bs, psi_h, psi_w, psi_ch = tf.shape(psi)
        psi = Adaptive_Upsampling2D(size=(xl_h, xl_w))(psi)
        
        # W_psi = BatchNormalization()(W_psi)
        
 
        return Multiply()([xl, psi])
    
class Adaptive_Upsampling2D_v2(Layer):
    
    def __init__(self, size=(2, 2), interpolation='bilinear', name='adaptive_upsampling', **kwargs):
        super(Adaptive_Upsampling2D_v2,self).__init__(**kwargs) 
        self.size = size
        if interpolation not in {'nearest', 'bilinear'}:
            raise ValueError('`interpolation` argument should be one of `"nearest"` '
                              'or `"bilinear"`.')
        self.interpolation = interpolation
    
    # @tf.function    
    def call(self, inputs):
        
        return tf.image.resize(inputs,[self.size[0], self.size[1]], method=self.interpolation)



class Attention_gate_v2(Layer):

    def __init__(self, F_int = None, name='attention_gate', **kwargs):
        super(Attention_gate_v2,self).__init__(name=name, **kwargs)
        self.F_int = F_int
        
    def build(self,input_shape):
        
        self.conv_W_xl = Conv2D(self.F_int, kernel_size=(1,1),strides=(1,1),
                                padding='valid',name=self.name+'/conv_W_xl') #theta
        self.conv_W_xg = Conv2D(self.F_int, kernel_size=(1,1),strides=(1,1),
                                padding='valid',name=self.name+'/conv_W_xg') #phi
        self.conv_W_psi = Conv2D(1, kernel_size=1,strides=1,
                                 padding='valid',name=self.name+'/conv_W_psi') #psi
    
    @tf.function
    def call(self, xl, xg):
     
        xl_h, xl_w = tf.shape(xl)[1], tf.shape(xl)[2]
        
        theta = self.conv_W_xl(xl)
        # theta = BatchNormalization()(theta)
        theta_h, theta_w = tf.shape(theta)[1], tf.shape(theta)[2]

        phi = self.conv_W_xg(xg)
        phi = Adaptive_Upsampling2D_v2(size=(theta_h, theta_w),
                                       name = self.name+'/adaptive_upsampling_1')(phi)
        # W_xg = BatchNormalization()(W_xg)

        psi = Add(name = self.name+'/Add')([theta,phi])
        psi = Activation('relu',name = self.name+'/relu')(psi)
        psi = self.conv_W_psi(psi)
        psi = Activation('sigmoid',name = self.name+'/sigmoid')(psi)
      
        psi = Adaptive_Upsampling2D_v2(size=(xl_h, xl_w),
                                       name = self.name+'/adaptive_upsampling_2')(psi)
        # W_psi = BatchNormalization()(W_psi)
        
        return Multiply(name = self.name+'/Multiply')([xl, psi])

    def get_config(self):
        config = super(Attention_gate_v2,self).get_config()
        config.update({'F_int':self.F_int,})
        return config
