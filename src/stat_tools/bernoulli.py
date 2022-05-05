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
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.distributions import distribution

from util.hparams import *
from layers import LAYER
import copy

tfd = tfp.distributions


__all__ = [
           'Bernoulli',
           ]

class Bernoulli(distribution.Distribution,tf.keras.layers.Layer):
    """ 
        *** Bernoulli Layer  ***
        
        Args:
            shape:                 The shape of samples.
            hparams:               The distributional parameters.
            network_hparams:       Network hyper-parameters for the module generating the distributional parameters.
            validate_args:         Python `bool`, default `False`. When `True` distribution parameters are checked for validity despite possibly degrading runtime
                                   performance. When `False` invalid inputs may silently render incorrect outputs.
            allow_nan_stats:       Python `bool`, default `True`. When `True`, statistics (e.g., mean, mode, variance) use the value "`NaN`" to indicate theresult is undefined. When `False`, an exception is raised if one or
                                   more of the statistic's batch members are undefined.
            dtype:                 The type of the event samples. `None` implies no type-enforcement.
            name:                  Python `str` name prefixed to Ops created by this class. Default: subclass name.               
    """
    
    
    def __init__(self,
                 shape,
                 hparams=None,
                 network_hparams=None,
                 validate_args=False,
                 allow_nan_stats=True,
                 dtype=tf.float32,
                 name='conditional_bernoulli',
                 ):
    
    
        # get default value for the hparams not defined
        if hparams is None: hparams=self.get_default_hparams()

        
        super(Bernoulli, self).__init__(reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
                                        validate_args=validate_args,
                                        allow_nan_stats=allow_nan_stats,
                                        dtype=dtype,
                                        name=name,)

 
        # set the regularizer for the network responsible for generating the distributional parameters.

        
        self._hparams=copy.deepcopy(hparams)
        
        self._network_hparams=copy.deepcopy(network_hparams)
        
        self._shape=shape
            

        self._network_hparams['filters']=1

        self._params_network = LAYER[self._network_hparams.layer_type](hparams=self._network_hparams)
                                                               
    def call(self,inputs,training):
        """"
            It calls the layer:
                i)  It calls the network to generate the distributional parameters.
                ii) It forms the distribution.
                
            Args:
                inputs:     Tensor with the conditioning factor of the distribution.
                training:   Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
            
        
        """
        
        with tf.name_scope(self.name or 'BernoulliLayer_call'):


            inputs = tf.convert_to_tensor(inputs, dtype=self.dtype, name='inputs')
            
            # call network
            self._params=self._params_network(inputs,training=training)
            
            
            # create Benroulli distribution
            self._p = tfp.distributions.Bernoulli(logits=self._params)            
            return 0.0

    
    def sample(self,condition,training):
        """ 
            It draws samples from the Bernoulli.
                Args:
                    condition: Tensor that contains the conditioning factor of the distribution.
                    training:  Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                      
        
                Returns:
                    The sample, a tensor of the shape: [batch_size,]+self._shape
                
        """
        
        with tf.name_scope(self.name or 'Bernoulli_sample'):
            self.call(inputs=condition,training=training)
            return  self._p.sample()
    
    def log_prob(self,x,condition,training=True,crop_size=0):
        """
        It computes the log-likelihood of the samples.
            Args:
                x:         Tensor that contains the samples whose log-likleihood to be computed. It should have shape: [batch_size,]+self._shape
                condition: Tensor that contains the conditioning factor of the distribution. The batch dimension should match that of x.
                training:  Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                crop_size: Exclude marginal pixels.  
            Returns:
                Tensor of shape [batch_size, ] containing the log probability.

        """

        with tf.name_scope(self.name or 'Bernoulli_log_prob'):
            
            x=tf.reshape(x, prefer_static.concat([[-1], self._shape], axis=0))
            
            self.call(inputs=condition,training=training)
            
            
            logp=self._p.log_prob(x)
               
            logp=logp[:,crop_size:-crop_size,crop_size:-crop_size,:]
             
            return tf.reduce_sum(logp,axis=[-1-i for i in range(tensorshape_util.rank(self._shape))])
        

    
    # Default hyperparameters of the logits network of the Bernoulli layer.
    @staticmethod
    def get_default_hparams():
        return HParams()




