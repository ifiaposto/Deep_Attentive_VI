# ============================================================================
#  Copyright 2021. 
#
#
#  Author: Ifigeneia Apostolopoulou 
#  Contact: ifiaposto@gmail.com, iapostol@andrew.cmu.edu 
#
#
# All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================




"""
    ***************************************************
    
    ****** Deep Attentive Variational Inference. ******
    
    ***************************************************
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow.compat.v2 as tf
import tensorflow.keras as tfk

from tensorflow.python.keras.layers import advanced_activations
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from util.hparams import *
import layers.resnet
from layers.util import DeepVaeInit
from layers.util import REGULARIZER


BN_EPS = 1e-5


__all__ = [
           'NonlocalResNetBlock',
           'DepthWiseAttention',
           'ALIGNMENT_FUNCTION',
           ]
        
class SoftmaxAlignment(tfk.layers.Layer):
    
    def __init__(self,
                 hparams,
                 key_dim=None,
                 use_scale=False,
                 **kwargs):
        
        super(SoftmaxAlignment, self).__init__(**kwargs)
        
        self._hparams=hparams
        self._key_dim=key_dim
        self._use_scale=use_scale
        
        self._built=False
        
        
      
    
    def build(self):
        
        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
       
        
        if self._use_scale:
            self.scale = self.add_weight(name='scale',
                                         shape=(),
                                         initializer=init_ops.ones_initializer(),
                                         regularizer=kernel_regularizer,
                                         dtype=self.dtype,
                                         trainable=True)
        else:
            self.scale=None
            
        self._softmax=advanced_activations.Softmax(axis=-1)
        
        self._built=True
            
    def call(self,proj_query,proj_key):
        
        ## [batch_size,key_dim,n]
        
        if not self._built:
            self.build()
        
        attention_scores = tf.linalg.matmul(proj_query, proj_key, transpose_a=True)
          
        if self.scale is not None:
            attention_scores *= self.scale
            
        ### column-wise normalization (each row is normalized). 
        
        ## attention_scores==\beta in [1]
        ## each row of attention_scores refers to the query
            
        
        return self._softmax(attention_scores)
    
    
class DepthWiseAttention(tf.Module):
    def __init__(self,
                 hparams,
                 attention_hparams,
                 ):
        
        self._key_dim=attention_hparams.key_dim
        self._query_dim=attention_hparams.query_dim
        self._use_scale=attention_hparams.use_scale
        alignment_function=attention_hparams.alignment_function
        self._hparams=hparams
        
        self._alignment_function=ALIGNMENT_FUNCTION[alignment_function](self._hparams,
                                                                        self._query_dim,
                                                                        self._use_scale)
        
    def apply(self,query,keys,values):
        
        ## query: [batch_size, width, height, query_dim]
        ## values: [batch_size, n, width, height, channels]
        ## keys: ## [batch_size, n, width, height, key_dim]
        
        ## [batch_size, width, height, key_dim,1]
        query=tf.expand_dims(query,axis=-1)
        
        ## [batch_size,  width, height, key_dim, n]
        keys=tf.keras.layers.Permute(dims=(2,3,4,1))(keys)
        
        
        ## [batch_size, width, height, 1, n]
        attention_scores=self._alignment_function.call(query, keys)
        ## [batch_size, width, height, n]
        attention_scores=attention_scores[:,:,:,0,:]
        
        
        ## [batch_size, width, height, channels, n]
        values=tf.keras.layers.Permute(dims=(2,3,4,1))(values)
        
        width=tf.shape(values)[1]
        height=tf.shape(values)[2]
        filters=tf.shape(values)[3]
        
        values=tf.squeeze(tf.linalg.matvec(values,attention_scores))
        values=tf.reshape(values,[-1,width,height,filters])
        
        return values

    @staticmethod
    def get_default_hparams():
        """
         default hyperparameters for the nonlocal-operation.
        """
        return HParams(key_dim=20,                    # key dimension
                       query_dim=20,                  # query dimension
                       value_dim=20,                  # context dimension
                       alignment_function='softmax',  # attention distribution
                       use_scale=True,                # flag indicating the attention scores should be scaled
                       use_layer_norm=False,          # flag indicating attention normalization
                       )
        
        
class NonlocalResNetBlock(tfk.layers.Layer):
    """
        Block that applies spatial, non-local, operations on 4D tensors as 
        described in [1].
        
        References:
             [1]. Wang, X., Girshick, R., Gupta, A. and He, K., 2018. Non-local neural networks. 
                 In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7794-7803).
    """
    
    def __init__(self,
                 hparams,
                 nonlocop_hparams,
                 **kwargs):
            
        super(NonlocalResNetBlock, self).__init__(**kwargs)
        
        
        self._key_dim=nonlocop_hparams.key_dim
        
        self._use_scale=nonlocop_hparams.use_scale
                
        self._alignment_function=nonlocop_hparams.alignment_function
        
        self._subsample=nonlocop_hparams.subsample
        
        self._hparams=hparams
        
        self._nonlocop_hparams=nonlocop_hparams
        
      
        
    def build(self, input_shape):
        
        if self._key_dim is None:
            self._key_dim=int(input_shape[-1]/8)
            
        
        Conv2D =  layers.resnet.weightnorm_layer(tf.keras.layers.Convolution2D,self._hparams.use_weight_norm,self._hparams.use_data_init)
        
        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
        
        bias_regularizer=REGULARIZER[self._hparams.bias_regularizer](l=self._hparams.lambda_reg) if self._hparams.bias_regularizer else None
        
        
        self.query_conv = Conv2D(filters = self._key_dim , 
                                 kernel_size= 1,
                                 use_bias=self._hparams.use_bias,
                                 kernel_initializer=DeepVaeInit(),
                                 bias_initializer=DeepVaeInit(),
                                 kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer,)
        
        
        self.key_conv =   Conv2D(filters = self._key_dim , 
                                 kernel_size= 1,
                                 use_bias=self._hparams.use_bias,
                                 kernel_initializer=DeepVaeInit(),
                                 bias_initializer=DeepVaeInit(),
                                kernel_regularizer=kernel_regularizer,
                                bias_regularizer=bias_regularizer,)
        
        
        self.value_conv =  Conv2D(filters =input_shape[-1] , 
                                  kernel_size= 1,
                                  use_bias=self._hparams.use_bias,
                                  kernel_initializer=DeepVaeInit(),
                                  bias_initializer=DeepVaeInit(),
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,)
        
 
        self._alignment_function=ALIGNMENT_FUNCTION[self._alignment_function](self._hparams,
                                                              self._key_dim,
                                                              self._use_scale)
  
        self.gamma = self.add_weight(name='gamma',
                                     shape=(),
                                     initializer=init_ops.zeros_initializer(),
                                     dtype=self.dtype,
                                     regularizer=self._hparams.kernel_regularizer,
                                     trainable=True)
        
        if self._nonlocop_hparams.use_layer_norm:

            self._attention_layer_norm_layer_a=tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                                gamma_regularizer=self._nonlocop_hparams.layer_norm_regularizer,
                                                                                beta_regularizer=self._nonlocop_hparams.layer_norm_regularizer)
            
            self._attention_layer_norm_layer_b=tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                                  gamma_regularizer=self._nonlocop_hparams.layer_norm_regularizer,
                                                                                  beta_regularizer=self._nonlocop_hparams.layer_norm_regularizer)
            

                                
                
            self._ln_conv2d=Conv2D(filters =input_shape[-1] , 
                                  kernel_size= 1,
                                  use_bias=self._hparams.use_bias,
                                  kernel_initializer=DeepVaeInit(),
                                  bias_initializer=DeepVaeInit(),
                                  kernel_regularizer=kernel_regularizer,
                                  bias_regularizer=bias_regularizer,)
                                                          
        
    def call(self, inputs):    
        """ check Generative Pretraining from Pixels"""
        
        keep_orig_inputs=self._nonlocop_hparams.use_layer_norm or self._subsample
        
        if keep_orig_inputs:
            orig_inputs=inputs
        
        if self._nonlocop_hparams.use_layer_norm:
            inputs=self._attention_layer_norm_layer_a(inputs)
        
        ## channels fiirst
    
        
        batch_size=tf.shape(inputs)[0]
        
        channels=tf.shape(inputs)[-1]
        
        width=tf.shape(inputs)[1]
        
        
        height=tf.shape(inputs)[2]
        
        ## [batch_size,width,height,key_dim]
        
        proj_query=self.query_conv(inputs)
        
        
        ## [batch_size,key_dim,width,height]
        
        ## channels first
        
        
        proj_query=tf.keras.layers.Permute(dims=(3,1,2))(proj_query)
        
        
        proj_query=tf.reshape(proj_query,[batch_size,self._key_dim,width*height])
        
        if self._subsample:
            inputs=tf.keras.layers.MaxPool2D(pool_size=(2, 2))(inputs)
            width=int(width/2)
            height=int(height/2)
    
        
        ## n in [1] is: n=width*height, d== # channels
        ## C_bar=key_dim
        
        ## proj_query==g(x)=Wg*x
    
        
        ## N is width*height
        
        ## [batch_size,key_dim,width*height]
    
        
        ## N is width*height
        
        ## [batch_size,key_dim,width*height]
        
       
        
        proj_key =  self.key_conv(inputs)
        
        ## [batch_size,key_dim,width,height]
        
        proj_key=tf.keras.layers.Permute(dims=(3,1,2))(proj_key)
        
        ## [batch_size,key_dim,width*height]
        
        ## proj_key==f(x)=Wf*x

        proj_key=tf.reshape(proj_key,[batch_size,self._key_dim,width*height])     
        
        attention_scores=self._alignment_function.call(proj_query, proj_key)
        
        ## [batch_size,width,height,channels]
    
        proj_value = self.value_conv(inputs)
        
        ## [batch_size,channels,width,height]
        
        proj_value=tf.keras.layers.Permute(dims=(3,1,2))(proj_value)
        ## [batch_size,channels,width*height]
        
        proj_value=tf.reshape(proj_value,[batch_size,channels,width*height])
        
        attention_output = math_ops.matmul(proj_value, attention_scores, transpose_b=True)
        
        if self._subsample:
            width=width*2
            height=height*2
        
        
        attention_output=tf.reshape(attention_output,[batch_size,channels,width,height])
        
        
        
        attention_output = tf.keras.layers.Permute(dims=(2,3,1))(attention_output)
        
        if not self._nonlocop_hparams.use_layer_norm:
            
            ## not-subsampled attention
            if not keep_orig_inputs:
        
                return self.gamma*attention_output +  inputs
            
            ## subsampled attention
            return self.gamma*attention_output +  orig_inputs
      
    
        attention_output_norm=self._ln_conv2d(self._attention_layer_norm_layer_b(self.gamma*attention_output +  orig_inputs))
            
            
        return orig_inputs+attention_output_norm
        
    
                       
    @staticmethod
    def get_default_hparams():
        """
         default hyperparameters for the nonlocal-operation.
        """
        return HParams(key_dim=32,                  # key dimension
                       query_dim=32,                # query dimension
                       use_scale=True,              # flag indicating the attention scores should be scaled
                       value_dim=32,                # context dimension
                       num_heads=1,                 # nof attention heads
                       subsample=False,             # flag indicating the spatial dimension should be subsampled
                       alignment_function='softmax',# attention distribution
                       use_layer_norm=False,        # flag indicating attention normalization
                       layer_norm_regularizer='l2', # regularizer function applied to the layer normalization weights
                )
        
ALIGNMENT_FUNCTION={"softmax": SoftmaxAlignment,
                    }
