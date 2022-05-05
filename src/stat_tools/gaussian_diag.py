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

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.distributions import distribution


from util.hparams import HParams
from layers import LAYER

tfd = tfp.distributions

import copy

__all__ = [
           'GaussianDiag',
           ]

class GaussianDiag(distribution.Distribution,tf.keras.layers.Layer):
    
    """ 
        *** Diagonal Gaussian Layer  ***
        
        It supports residual parametrization, as introduced in [1].
        References:
        
        [1] Vahdat A, Kautz J. Nvae: A deep hierarchical variational autoencoder. Advances in Neural Information Processing Systems. 2020
        
        
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
                 name="gaussian_diag"
                 ):

        super(GaussianDiag, self).__init__(dtype=dtype,
                                           reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                                           validate_args=validate_args,
                                            allow_nan_stats=allow_nan_stats,
                                           name=name,
                                           )

        self._shape=shape

        self._params_network=None
        
        self._hparams=copy.deepcopy(hparams)
        
        self._network_hparams=copy.deepcopy(network_hparams)
        
        self.build()

        
    def build(self):
        
        # build the network of the parameters
        
        shape_l=self._shape.as_list()
        
        # double output since the network should generate both the mean and the log-scale.
        shape_l[-1]=2*shape_l[-1]
                
        self._network_hparams['filters']=shape_l[-1]
                
        self._params_network = LAYER[self._network_hparams.layer_type](hparams=self._network_hparams)
            
    def call(self, condition, training ,initial_params=None):

        """
            It calls the layer:
                i)  It calls the network to generate the distributional parameters.
                ii) It forms the distribution.
                
            Args:
                inputs:         Tensor with the conditioning factor of the distribution.
                training:       Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                initial_params: If not 'None', the initial parameters of the residual update. 
            
        """
        with tf.name_scope(self.name or 'GaussianLayerDiag_call'):
            
            condition = tf.convert_to_tensor(condition, dtype=self.dtype, name='gaussian_condition')
        
            self._delta_mu=[]
            self._delta_sigma_diag=[]
            
            self._mu=[]
            self._sigma_diag=[]
            

            ## insert an offset mean and log-sigma in the parameters.
            if initial_params is not None:
               
        
                mu,log_sigma=tf.split(initial_params,2,axis=-1)

   
                self._delta_mu.append(mu)
                self._delta_sigma_diag.append(log_sigma)
                self._mu.append(mu)
                self._sigma_diag.append(log_sigma)

            ## call the network parameters and extract the tensors for the mean and log_sigma
            params=self._params_network(condition,training=training)

            mu,log_sigma=tf.split(params,2,axis=-1)
            

            self._delta_mu.append(mu)

            if self._hparams.noise_stddev>0:
                log_sigma=tf.keras.layers.GaussianNoise(self._hparams.noise_stddev)(log_sigma)


            self._delta_sigma_diag.append(log_sigma)
        
            ## create the final parameters
            self.create_params()
                            
            ## smooth thresholding, see: ## https://en.wikipedia.org/wiki/Smoothstep#Variations
            def bound_param(x,mi,mx):
                if abs(mi)!=abs(mx):
                    ## manually transpose to suppress the bug: layout failed: Invalid argument: size of values 0 does not match size of permutation 4.
                    ## https://github.com/tensorflow/tensorflow/issues/34499
                    x = tf.transpose(x, [0, 3, 1, 2])
                    x= mi + (mx-mi)*(lambda t: tf.where(t < 0.0 , 0.0, tf.where( t <= 1.0 , 3*tf.math.pow(t,2)-2*tf.math.pow(t,3), 1.0 ) ) )( (x-mi)/(mx-mi) )
                    return tf.transpose(x, [0, 2, 3, 1])
                return abs(mi)*tf.math.tanh(x/(2*abs(mi)))
            
            ## bound the distributional parameters to ease stability.
            self._p = [tfd.MultivariateNormalDiag(loc=bound_param(self._mu[i],-5.0,5.0),scale_diag=tf.math.exp(bound_param(self._sigma_diag[i],self._hparams.log_scale_low_bound,self._hparams.log_scale_upper_bound))+1e-2) for i in range(len(self._mu))]


            # check the reparametrization trick, by Kingma, can be applied
            for p in self._p:
                assert p.reparameterization_type == tfd.FULLY_REPARAMETERIZED
                


            return 0.0


            
    def create_params(self):
        """
            It forms the shift and scale of the Gaussian.
        """
        
        offset=len(self._delta_mu)-1

         
        for i in range(offset,1+offset):
          
            if i==0:

                self._sigma_diag.append(self._delta_sigma_diag[i])
                self._mu.append(self._delta_mu[i])
            else:
                ## residual Gaussian distribution
                mu=tf.math.add(self._mu[i-1],self._delta_mu[i])

                self._mu.append(mu)
                
     
                sigma_diag=tf.math.add(self._delta_sigma_diag[i],self._delta_sigma_diag[i-1])
            
                self._sigma_diag.append(sigma_diag)

        
    def sample(self,condition,training=True,num_samples=1,initial_params=None):
        """ 
            It draws samples from the Gaussian.
                Args:
                    condition:         Tensor that contains the conditioning factor of the distribution.
                    training:          Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                    initial_params:    If not 'None', the initial parameters of the residual update. 
                Returns:
                    The sample, a tensor of the shape: [batch_size,]+self._shape
        """

        sample_shape = tf.cast([num_samples], tf.int32, name='sample_shape')
        with tf.name_scope(self.name or 'GaussianLayerDiag_sample'):
            
            self.call(condition=condition,training=training,initial_params=initial_params)
            sample_shape=sample_shape
            return self._p[-1].sample(sample_shape=sample_shape)
    
    def log_prob(self,x,condition,training=True,initial_params=None,compute_dist=True):
        """
        It computes the log-likelihood of the samples.
            Args:
                x:               Tensor that contains the samples whose log-likleihood to be computed. It should have shape: [batch_size,]+self._shape
                condition:       Tensor that contains the conditioning factor of the distribution. The batch dimension should match that of x.
                training:        Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                initial_params:  If not 'None', the initial parameters of the residual update. 
                compute_dist:    The network of the distribution is called from scratch.
            Returns:
                Tensor of shape [batch_size, ] containing the log probability.
        """
        with tf.name_scope(self.name or 'GaussianLayerDiag_log_prob'):
            x = tf.convert_to_tensor(x, dtype=self.dtype, name='inputs')
 
            if x.shape[-1]!=self._shape:
                x=tf.reshape(x, prefer_static.concat([[-1], self._shape], axis=0))
                
            if compute_dist:
                self.call(condition=condition,training=training,initial_params=initial_params)
            
            return tf.reduce_sum(self._p[-1].log_prob(x),axis=[-1-i for i in range(tensorshape_util.rank(self._shape)-1)])


    
    def params(self,network=-1):
        """ Utility function that merges the parameters in a single tensor """
        return tf.concat([self._mu[network],self._sigma_diag[network]],axis=-1)


    # Default hyperparameters of  the Gaussian layer.
    @staticmethod
    def get_default_hparams():
        """ default parameters for the Gaussian """
        return HParams(
                       # lower bound for the log-scale, to ease stability.
                       log_scale_low_bound=-5.0,
                       # upper bound for the log-scale, to ease stability.
                       log_scale_upper_bound=5.0,
                       # Gaussian noise regularization to be applied to the log-scale, to ease generalization.
                       noise_stddev=-1.0,
                       )



