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

import tensorflow.compat.v2 as tf
import tensorflow.keras as tfk

import copy
from tensorflow_probability.python.internal import prefer_static
from layers.weight_norm import weightnorm_layer
from util.hparams import HParams
from layers.util import ACTIVATION,REGULARIZER,swish,DeepVaeInit
from layers.attention import NonlocalResNetBlock

# batch normalization, epsilon constant
BN_EPS = 1e-5



__all__ = [
           'ResNetBlock',
           'ResNet',
           'Conv2D',
           'ConcatConv2D',
           'Conv2DSum',
           'ResNetDecoder',
           'FactorizedReduce',
           'SqueezeAndExciteBlock',
           ]
     
class FactorizedReduce(tfk.layers.Layer):
    
    def __init__(self,
                 num_filters,
                 hparams,
                  **kwargs):
        
        super(FactorizedReduce, self).__init__(**kwargs)
        
        self._num_filters=num_filters
        
        self._hparams=hparams
    
            
    def build(self, input_shape):
         

        Conv2D =  weightnorm_layer(tf.keras.layers.Convolution2D,self._hparams.use_weight_norm,self._hparams.use_data_init)
        
        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
        
        bias_regularizer=REGULARIZER[self._hparams.bias_regularizer](l=self._hparams.lambda_reg) if self._hparams.bias_regularizer else None
           
        self.conv_1  =Conv2D(filters=self._num_filters//4,
                                        kernel_size=1,
                                        activation=None,
                                        strides=2,
                                        use_bias=self._hparams.use_bias,
                                        kernel_initializer=DeepVaeInit(),#DeepVaeInit(),
                                        bias_initializer=DeepVaeInit(),
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        )
        self.conv_2  =Conv2D(filters=self._num_filters//4,
                                        kernel_size=1,
                                        activation=None,
                                        strides=2,
                                        use_bias=self._hparams.use_bias,
                                        kernel_initializer=DeepVaeInit(),
                                        bias_initializer=DeepVaeInit(),
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        )
        self.conv_3  =Conv2D(filters=self._num_filters//4,
                                        kernel_size=1,
                                        activation=None,
                                        strides=2,
                                        use_bias=self._hparams.use_bias,
                                        kernel_initializer=DeepVaeInit(),
                                        bias_initializer=DeepVaeInit(),
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        )
        self.conv_4 =Conv2D(filters= self._num_filters - 3 * (self._num_filters // 4),
                                        kernel_size=1,
                                        strides=2,
                                        activation=None,
                                        use_bias=self._hparams.use_bias,
                                        kernel_initializer=DeepVaeInit(),
                                        bias_initializer=DeepVaeInit(),
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        )
    def call(self, inputs):   
        
        out =swish(inputs)
        conv1 = self.conv_1(out)
        conv2 = self.conv_2(out[:,  1:, 1:,:])
        conv3 = self.conv_3(out[:,  :, 1:,:])
        conv4 = self.conv_4(out[:, 1:, :,:])
        
        out = tf.concat([conv1, conv2, conv3, conv4], axis=-1)
        
        return out 



class ResNetBlock(tfk.layers.Layer):
    
    """
        ResNet cell
        
        References:
        
            [1].He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition.
                In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
                The internal blocks can be either MLPs or ConNets.

        
        """
    def __init__(self,
                 scale=None,
                 kernel_size=3,
                 hparams=None,
                 num_nodes=2,
                 use_weight_norm=True,
                 use_batch_norm=True,
                 use_data_init=True,
                 node_type='conv2d',
                 **kwargs):
        
        super(ResNetBlock, self).__init__(**kwargs)
        
        # the hyperparameters for conv2d
        self._hparams=copy.deepcopy(hparams)
        
  
        ## whether it will down sample/ upsample the input or it will preserve the dimension
        self._scale=scale
        
        self._use_weight_norm=use_weight_norm
        
        self._use_data_init=use_data_init
        
        self._use_batch_norm=use_batch_norm
        
        self._num_nodes=num_nodes
        

        self._kernel_size=kernel_size
        
        self._node_type=node_type
    
            
        # To be built in `build`.
        self._nodes=[]

        self._shortcut=None
        self._dropout=None
    
        
    def build(self, input_shape):
        
        ## number of filters
        
        if self._scale<0:
            self._hparams['filters']=abs(self._scale)*input_shape[-1]
            self._hparams['strides']=abs(self._scale)
             
        elif self._scale>0:
            self._hparams['filters']=input_shape[-1]//2
            self._hparams['strides']=1
        else:
            self._hparams['filters']=input_shape[-1]
            self._hparams['strides']=1
        
        
        #### build the shortcut layer ####
        
        if self._scale ==0:
            self._shortcut=tfk.layers.Lambda(lambda x:x)
            
        elif self._scale<0:
            self._shortcut=FactorizedReduce(num_filters=self._hparams['filters'],
                                                  hparams=self._hparams, 
                                                )
            
            
        elif self._scale>0:
            dim=int(input_shape[1]*abs(self._scale))
            shape=[dim,dim]
            self._shortcut_hparams=copy.deepcopy(self._hparams)

            self._shortcut_hparams['use_bias']=False
            self._shortcut_hparams['activation']=''
            self._shortcut_hparams['kernel_size']=1
            self._shortcut_hparams['use_batch_norm']=False
            self._shortcut_hparams['use_se']=False
            
            
            self._shortcut=tf.keras.Sequential([tfk.layers.Lambda(lambda x:tf.compat.v1.image.resize_bilinear(x, shape, align_corners=True)),
                                                      Conv2D(hparams=self._shortcut_hparams,)
                                                      ])
        
        self._nodes=[]
        
        node_hparams=copy.deepcopy(self._hparams)
        if self._num_nodes==1:
            node_hparams['use_se']=self._hparams.use_se
        else:
            node_hparams['use_se']=False

        
        #### build first node that performs upsamling/downsampling if needed
        if self._scale==0 or self._scale<0:
           
            self._nodes.append(LAYER[self._node_type](hparams=node_hparams))
        else:
            dim=int(input_shape[1]*abs(self._scale))
            shape=[dim,dim]
            self._nodes.append(tf.keras.Sequential([tfk.layers.Lambda(lambda x:tf.compat.v1.image.resize_nearest_neighbor(x, shape, align_corners=False)),
                                                    LAYER[self._node_type](hparams=node_hparams,)
                                                    ]))
                                                    
        ### build the remaining nodes
        for i in range(self._num_nodes-1):
            node_hparams=copy.deepcopy(self._hparams)
            node_hparams['strides']=1
            
            if i==self._num_nodes-2:
                node_hparams['use_se']=self._hparams.use_se
            else:
                node_hparams['use_se']=False
            
                
            self._nodes.append(LAYER[self._node_type](hparams=node_hparams,))

        # Record that the layer has been built.
        super(ResNetBlock, self).build(input_shape)
        
        
    def call(self, inputs, training):
            with tf.name_scope(self.name or 'ResNetBlock_call'):
               
                inputs = tf.convert_to_tensor(inputs, dtype=self.dtype, name='resnet_inputs')

                shortcut = self._shortcut(inputs)
                
                temps=inputs
                for i in range(self._num_nodes):
                    temps=self._nodes[i](temps,training)
                
                residual=tf.math.scalar_mul(0.1,temps)

                return tf.keras.layers.Add()([residual, shortcut])
            
class SqueezeAndExciteBlock(tfk.layers.Layer):
    """
    Squeeze and Excite Blocks
    
    References:
    
    [1]. Hu, J., Shen, L. and Sun, G., 2018. Squeeze-and-excitation networks. 
        In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 7132-7141).
    """
    def __init__(self,
                 hparams,
                 **kwargs):

        super(SqueezeAndExciteBlock, self).__init__(**kwargs)

        self._hparams=hparams
        
        if hasattr(self._hparams, 'se_ratio'):
            self._ratio=self._hparams.se_ratio
        else:
            self._ratio=16

        self._use_weight_norm=self._hparams.use_weight_norm

        self._use_data_init=self._hparams.use_data_init


   
            
    def build(self, input_shape):

        self._filters=input_shape[-1]

        Dense =   weightnorm_layer(tf.keras.layers.Dense,self._use_weight_norm,self._use_data_init)
        
        
               
        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
        

        self._first_dense_layer=Dense(units=max(self._filters//self._ratio,4),
                                      activation='relu',
                                      kernel_initializer=DeepVaeInit(),
                                      kernel_regularizer=kernel_regularizer,
                                      use_bias=False,
                                      )

        self._second_dense_layer=Dense(units=self._filters,
                                       activation='sigmoid',
                                       kernel_initializer=DeepVaeInit(),
                                       kernel_regularizer=kernel_regularizer,
                                       use_bias=False,
                                       )
        
    def call(self, inputs):
        with tf.name_scope(self.name or 'SqueezeAndExcite_call'):

            inputs = tf.convert_to_tensor(inputs, dtype=self.dtype, name='squeeze_and_excite_inputs')

            se_inputs =  tf.keras.layers.GlobalAveragePooling2D()(inputs)

            se_shape = tf.TensorShape([1, 1, self._filters])

            se_inputs=tf.reshape(se_inputs,shape=prefer_static.concat([[-1],se_shape ], axis=0))

            se_inputs=self._first_dense_layer(se_inputs)

            se_inputs=self._second_dense_layer(se_inputs)

            se_inputs=tf.math.multiply(inputs,se_inputs)

            return se_inputs
            
class ConcatConv2D(tfk.layers.Layer):
    def __init__(self,
                 hparams=None,
                 num_filters=None,
                 **kwargs):
      
        super(ConcatConv2D, self).__init__(**kwargs)
          
        self._hparams=hparams
        
        self._num_filters=num_filters
          
        self._conv2d_layer=None
          
    def build(self, input_shape):
          
        Conv2D =  weightnorm_layer(tf.keras.layers.Convolution2D,self._hparams.use_weight_norm,self._hparams.use_data_init)
        
        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
        
        bias_regularizer=REGULARIZER[self._hparams.bias_regularizer](l=self._hparams.lambda_reg) if self._hparams.bias_regularizer else None
        
    
        
        self._num_filters=input_shape[0][-1] if self._num_filters is None else self._num_filters
        
        self._conv2d_layer=Conv2D(filters=self._num_filters ,
                                 kernel_size=self._hparams.kernel_size,
                                 kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 kernel_initializer=DeepVaeInit(),
                                 bias_initializer=DeepVaeInit(),
                                 padding='same',
                                 strides=1,
                                 )
        
        if self._hparams.use_nonlocal:
            self._nonlocal_layer=NonlocalResNetBlock(hparams=self._hparams,
                                                     nonlocop_hparams=self._hparams.nonlocop_hparams
                                                     )

        
        
    def call(self, inputs):
        with tf.name_scope(self.name or 'ConcatConv2D_call'):
              
            temps_a = tf.convert_to_tensor(inputs[0], dtype=self.dtype, name='concat combiner inputs a')
              
            temps_b = tf.convert_to_tensor(inputs[1], dtype=self.dtype, name='concat combiner inputs b')
              
            temps_concat= tf.concat([temps_a,temps_b],axis=-1)       
              
            res= self._conv2d_layer(temps_concat)
            
            
            if self._hparams.use_nonlocal:
                res=self._nonlocal_layer(res)
            
            return res
         
    @staticmethod
    def get_default_hparams():
            """
                  default hyperparameters.
             
            """
            return HParams(layer_type='concat_conv2d',
                           kernel_size=1,
                           use_bias=True,
                           kernel_regularizer='l2',
                           bias_regularizer='l2',
                           lambda_reg=1.5e-4,
                           use_weight_norm=True,
                           use_data_init=False,
                           
                           use_nonlocal=False,
                           nonlocop_hparams=NonlocalResNetBlock.get_default_hparams(),
                           )
            
             
class Conv2DSum(tfk.layers.Layer):
    def __init__(self,
                 hparams=None,
                 num_filters=None,
                 **kwargs):
        super(Conv2DSum, self).__init__(**kwargs)
         
        self._hparams=hparams
         
        self._conv2d_layer=None
        
        self._num_filters=num_filters
          
    def build(self, input_shape):
         
        Conv2D =  weightnorm_layer(tf.keras.layers.Convolution2D,self._hparams.use_weight_norm,self._hparams.use_data_init)
        
        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
        
        bias_regularizer=REGULARIZER[self._hparams.bias_regularizer](l=self._hparams.lambda_reg) if self._hparams.bias_regularizer else None
        
        
        self._num_filters=input_shape[0][-1] if self._num_filters is None else self._num_filters 
         
        self._conv2d_layer=Conv2D(filters=self._num_filters,
                                 kernel_size=self._hparams.kernel_size,
                                 kernel_regularizer=kernel_regularizer,
                                 bias_regularizer=bias_regularizer,
                                 kernel_initializer=DeepVaeInit(),
                                 bias_initializer=DeepVaeInit(),
                                 padding='same',
                                 strides=1,
                                 )
        
        if self._num_filters!=input_shape[0][-1]:
            self._conv2d_layer_a=Conv2D(filters=self._num_filters ,
                                        kernel_size=self._hparams.kernel_size,
                                        kernel_regularizer=kernel_regularizer,
                                        bias_regularizer=bias_regularizer,
                                        kernel_initializer=DeepVaeInit(),
                                        bias_initializer=DeepVaeInit(),
                                        padding='same',
                                        strides=1,
                                        )
        else:
            self._conv2d_layer_a=tf.keras.layers.Lambda(lambda x:tf.concat(x,axis=-1))
            

        if self._hparams.use_nonlocal:
            self._nonlocal_layer=NonlocalResNetBlock(hparams=self._hparams,
                                                     nonlocop_hparams=self._hparams.nonlocop_hparams
                                                     )
        
        
    def call(self, inputs):
        with tf.name_scope(self.name or 'Conv2DSum_call'):
             
            temps_a = tf.convert_to_tensor(inputs[0], dtype=self.dtype, name='sum combiner inputs a')
             
            temps_b = tf.convert_to_tensor(inputs[1], dtype=self.dtype, name='concat combiner inputs b')
            
            
                 
            temps_b=self._conv2d_layer(temps_b)
            
            temps_a=self._conv2d_layer_a(temps_a)
            
            
            res=tf.math.add(temps_a,temps_b)
            
            if self._hparams.use_nonlocal:
                
                res=self._nonlocal_layer(res)

             
            return  res
         
    @staticmethod
    def get_default_hparams():
            """
                  default hyperparameters.
             
            """
            return HParams(layer_type='conv2d_sum',
                           kernel_size=1,
                           use_bias=True,
                           kernel_regularizer='l2',
                           bias_regularizer='l2',
                           lambda_reg=1.5e-4,
                           use_weight_norm=True,
                           use_data_init=False,
                           use_nonlocal=False,
                           nonlocop_hparams=NonlocalResNetBlock.get_default_hparams()
                           )
            
            
            
class Conv2D(tfk.layers.Layer):
    def __init__(self,
                 hparams=None,
                 **kwargs):
                
        super(Conv2D, self).__init__(**kwargs)
                
        self._hparams=hparams
        
        self._num_times=self._hparams.num_times if hasattr(self._hparams, 'num_times') else 1
        
        self._conv2d_layers=[None]*self._num_times
        
        
        if hparams.activation:
            kwargs = {key:getattr(hparams, 'activation_'+key) for key in ACTIVATION[self._hparams.activation].__code__.co_varnames[1:]}
            self._activation=lambda x: ACTIVATION[self._hparams.activation](x)
        else:
            self._activation=lambda x:x
            

                            
    def build(self, input_shape):
        
        Conv2D =  weightnorm_layer(tf.keras.layers.Convolution2D,self._hparams.use_weight_norm,self._hparams.use_data_init)

        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
        
        bias_regularizer=REGULARIZER[self._hparams.bias_regularizer](l=self._hparams.lambda_reg) if self._hparams.bias_regularizer else None
        

       
        
        if self._hparams.use_batch_norm:
            
          
            batch_norm_regularizer=REGULARIZER[self._hparams.batch_norm_regularizer](l=self._hparams.lambda_reg) if self._hparams.batch_norm_regularizer else None
        
 
            batch_norm_layer=tf.keras.layers.BatchNormalization
        
            self._batch_norm_layer=batch_norm_layer(momentum=self._hparams.bn_momentum,
                                                    epsilon=BN_EPS,
                                                    gamma_regularizer=batch_norm_regularizer,
                                                    beta_regularizer=batch_norm_regularizer)
        else:
            self._batch_norm_layer=tfk.layers.Lambda(lambda x:x)
        
        if self._hparams.filters<0:
            self._hparams['filters']=input_shape[-1]
                
        for i in range(self._num_times):
            
            self._conv2d_layers[i]=Conv2D(filters=input_shape[-1] if i<self._num_times-1 else self._hparams.filters,
                                          strides=self._hparams.strides,
                                          kernel_size=1 if i<self._num_times-1 else self._hparams.kernel_size,
                                          kernel_regularizer=kernel_regularizer,
                                          bias_regularizer=bias_regularizer,
                                          kernel_initializer=DeepVaeInit(),
                                          bias_initializer=DeepVaeInit(),
                                          use_bias=self._hparams.use_bias,
                                          padding='same',
                                          activation=None,
                                          )
            

        if self._hparams.use_se:
            self._se_layer=SqueezeAndExciteBlock(hparams=self._hparams)
            
        if self._hparams.use_nonlocal:
            self._nonlocal_layer=NonlocalResNetBlock(hparams=self._hparams,
                                                     nonlocop_hparams=self._hparams.nonlocop_hparams
                                                     )
        
        
    def call(self, inputs,training):
        
        with tf.name_scope(self.name or 'Conv2D_call'):

            temps = tf.convert_to_tensor(inputs, dtype=self.dtype, name='resnet_inputs') 
            
            temps=self._batch_norm_layer(temps,training)
            
            if self._hparams.use_nonlocal:
                temps=self._nonlocal_layer(temps)
            
            for i in range(self._num_times):  
                
                temps=self._activation(temps)
                temps=self._conv2d_layers[i](temps)


            if self._hparams.use_se:
                temps=self._se_layer(temps)
                
            return temps
        
    @staticmethod
    def get_default_hparams():
            """
                default hyperparameters.
            
            """
            return HParams(layer_type='conv2d',
                           kernel_size=3,
                           num_times=2,
                           strides=1,
                           use_bias=True,
                           kernel_regularizer='l2',
                           bias_regularizer='l2',
                           lambda_reg=1.5e-4,
                           use_se=False,
                           use_weight_norm=True,
                           use_data_init=False,
                           activation="",
                           activation_alpha=0.0,
                            # parameters for the activation function
                            activation_max_value=1e+3,
                            activation_threshold=0,                      
                            activation_kwargs='',
                            activation_result='',
                            # parameters for the batch normalization
                            use_batch_norm=False,
                            bn_momentum=0.9,
                            batch_norm_regularizer='l2',
                            # parameters for the non-local blocks
                            use_nonlocal=False,
                            nonlocop_hparams=NonlocalResNetBlock.get_default_hparams(),
                           )

class ResNet(tfk.layers.Layer):
    
    def __init__(self,
                 hparams=None,
                 **kwargs):
        """
            ResNet 
            
            References:
            
            [1].He, K., Zhang, X., Ren, S. and Sun, J., 2016. Deep residual learning for image recognition.
                In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
                The internal blocks can be either MLPs or ConNets.
            
        """
                 
                 
        super(ResNet, self).__init__(**kwargs)
        
        self._hparams=hparams
       
        self._use_stem=hparams.use_stem
        
        self._stem_layer=tf.keras.layers.Lambda(lambda x:x)
        

        if not isinstance(self._hparams.scale,list):
            self._hparams.scale=[self._hparams.scale]*self._hparams.num_blocks
            
        self._residual_blocks=[None]*self._hparams.num_blocks
            
    def build(self, input_shape):
        
        
        Conv2D =  weightnorm_layer(tf.keras.layers.Convolution2D,self._hparams.use_weight_norm,self._hparams.use_data_init)
    
        kernel_regularizer=REGULARIZER[self._hparams.kernel_regularizer](l=self._hparams.lambda_reg) if self._hparams.kernel_regularizer else None
        
        bias_regularizer=REGULARIZER[self._hparams.bias_regularizer](l=self._hparams.lambda_reg) if self._hparams.bias_regularizer else None
        
        
        if self._use_stem:
            
            self._stem_layer=Conv2D(filters=self._hparams.num_filters,
                                    kernel_size=self._hparams.stem_kernel_size,
                                    kernel_regularizer=kernel_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    kernel_initializer=DeepVaeInit(),
                                    bias_initializer=DeepVaeInit(),
                                    padding='same',)
            



        for k in range(self._hparams.num_blocks):


            self._residual_blocks[k]=ResNetBlock(kernel_size=self._hparams.kernel_size,
                                                 hparams=self._hparams,
                                                 scale=self._hparams.scale[k],
                                                 use_batch_norm=self._hparams.use_batch_norm,
                                                 use_weight_norm=self._hparams.use_weight_norm,
                                                 use_data_init=self._hparams.use_data_init,
                                                 node_type=self._hparams.node_type,
                                                 num_nodes=self._hparams.num_nodes,
                                                 )            

        # Record that the layer has been built.
        super(ResNet, self).build(input_shape)
        
 
    def call(self, inputs,training):
        
        with tf.name_scope(self.name or 'ResNet_call'):
        
            
            temps = tf.convert_to_tensor(inputs, dtype=self.dtype, name='resnet_inputs')

           
            temps= self._stem_layer(temps)
           
            for k in range(self._hparams.num_blocks):
                temps=self._residual_blocks[k](temps,training) 
            
                      
            return temps
               

                                            
    def save_hparams(self, csv_logger):
        csv_logger.writerow(['resnet  hyperparams'])
        self._hparams.save(csv_logger)
    
    def get_config(self):

                
                        
        config = super().get_config().copy()
        config.update({
                      'hparams': self._hparams.__dict__,
                      })
        return config
    
    @staticmethod
    def get_default_hparams():
        """
         default hyperparameters.
            
        """
        return HParams(layer_type='resnet',
                       node_type='conv2d',
                       num_nodes=2,                 # nof convolutions per module
                       use_stem=True,               # flag indicating the input will be intiially mapped with a convolution.
                       kernel_size=3,               # kernel size of convolutions
                       stem_kernel_size=3,          # kernel size of convolutions in stem block
                       num_blocks=2,                # nof resnet blocks
                       scale=-2,                    # list indicating whether downsampling (<0), upsampling (>0) will be applied, or the spatial dimension (==0) will be preserved
                       activation="relu",           # activation function
                       # parameters for the activation function
                       # refer to tensorflow documentation:
                       # https://www.tensorflow.org/api_docs/python/tf/keras/activations
                       activation_alpha=0.0,
                       activation_max_value=1e+3,
                       activation_threshold=0,                      
                       activation_kwargs='',
                       output_activation_kwargs='',
                       activation_result='',
                       num_filters=16,               # if use_stem=True, the nof channels of the step mapping
                       use_bias=True,                # flag indicating whether the layer uses a bias vector
                       kernel_regularizer='l2',      # regularizer function applied to the kernel weights matrix
                       bias_regularizer='l2',        # regularizer function applied to the bias vector
                       batch_norm_regularizer='l2',  # regularizer function applied to beta, gamma weights of batch normalization
                       lambda_reg=1.5e-4,            # regularization coefficient
                       use_se=False,                 # flag indicating whether squeeze-and-excite blocks will be used
                       use_weight_norm=True,         # flag indicating whether weight normalization will be applied
                       use_data_init=False,          # flag indicating whether data initialization will be applied
                       use_batch_norm=True,          # flag indicating whether batch-normalization will be applied
                       bn_momentum=0.9,              # batch normalization momentum
                       # attention related parameters
                       use_nonlocal=False,           # flag indicating whether non-local, spatial blocks will be used
                       ## parameters for the non-local blocks
                       nonlocop_hparams=NonlocalResNetBlock.get_default_hparams(),
                       )


LAYER = {"conv2d":Conv2D,
         "conv2d_sum":Conv2DSum,
         "concat_conv2d":ConcatConv2D,
         "resnet":ResNet,
}