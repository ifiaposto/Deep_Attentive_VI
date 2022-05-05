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
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp

import numpy as np
import copy

tfb = tfp.bijectors
tfd = tfp.distributions


from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization
from tensorflow_probability.python.distributions import distribution
from layers import LAYER
from util.hparams import HParams


tfd = tfp.distributions

__all__ = [
           'DiscretizedLogisticMixture',
           ]

class DiscretizedLogisticMixture(distribution.Distribution,tf.keras.layers.Layer):

    """
        *** Discretized logistic pixel distribution ***
        
        References:
        
        [1]. Salimans, T., Karpathy, A., Chen, X. and Kingma, D.P., 2017.
              Pixelcnn++: Improving the pixelcnn with discretized logistic mixture likelihood and other modifications.
              arXiv preprint arXiv:1701.05517.
        
        [2]. Oord, A.V.D., Kalchbrenner, N. and Kavukcuoglu, K., 2016.
             Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759.
        
        [3].  Van den Oord, A., Kalchbrenner, N., Espeholt, L., Vinyals, O. and Graves, A., 2016.
              Conditional image generation with pixelcnn decoders.
              In Advances in neural information processing systems (pp. 4790-4798).
              
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
                 name='conditional_logistic_mixture',
                 ):
    
        # get default value for the hparams not defined
        if hparams is None: hparams=self.get_default_hparams()



        self._log_scale_low_bound=tf.dtypes.cast(hparams.log_scale_low_bound, tf.float32)
        
        super(DiscretizedLogisticMixture, self).__init__(dtype=dtype,
                                                         reparameterization_type=reparameterization.NOT_REPARAMETERIZED,
                                                         validate_args=validate_args,
                                                         allow_nan_stats=allow_nan_stats,
                                                         name=name,
                                                         )
        
        self._hparams=copy.deepcopy(hparams)
        
        self._network_hparams=copy.deepcopy(network_hparams)
        
        self._shape=shape



        self._num_channels = self._shape[-1] if tensorshape_util.rank(self._shape)>1 else 1
        
        self._num_pixels=self._shape[:-1] if tensorshape_util.rank(self._shape)>1 else  [self._shape[-1]]
        
        self._num_params_total=prefer_static.concat([self._num_pixels,[self._hparams.num_logistic_mix*self._num_channels*2+self._hparams.num_logistic_mix*3+self._hparams.num_logistic_mix]],axis=0)
             

      
        self._network_hparams['filters']=self._num_params_total[-1]
        self._params_network = LAYER[self._network_hparams.layer_type](hparams=self._network_hparams)
               
    def call(self,inputs,training):
        """
            It calls the network to generate the distributional parameters.
                
                
            Args:
                inputs:     Tensor with the conditioning factor of the distribution.
                training:   Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
        
        """
      
        
        with tf.name_scope(self.name or 'DiscretizedLogisticMixtureLayer_call'):
                        
            inputs = tf.convert_to_tensor(inputs, dtype=self.dtype, name='inputs')
            

            self._params=self._params_network(inputs,training=training)
            

            



    ### 
    def create_params(self,x):
        """
            It extracts the locations, scales, logit probabilities and the channel regression coefficients from the single tensor-output of the parameters network.
            Args:
                x: 4D tensor containing the raw parameters.
        """

        params=self._params

          

        self._logits_probs=params[:,:,:,:self._hparams.num_logistic_mix]


        params=tf.reshape(params[:,:,:,self._hparams.num_logistic_mix:],prefer_static.concat([[-1],self._shape[:-1],[self._hparams.num_logistic_mix,2*self._shape[-1]+3]], axis=0))

        coeffs_i=tf.math.tanh(params[:,:,:,:,:3])

        coeffs_i=tf.keras.backend.permute_dimensions(coeffs_i,pattern=[0,1,2,4,3])


        params=tf.reshape(params[:,:,:,:,3:],prefer_static.concat([[-1],self._shape,[self._hparams.num_logistic_mix*2]], axis=0))

        self._locs, self._log_scales = tf.split(params, 2, axis=-1)

        xs=prefer_static.concat([[-1],self._shape,[1]],axis=0)

        x = tf.reshape(x, xs)
                

        x=tf.tile(x,multiples=tf.cast([1,1,1,1,self._hparams.num_logistic_mix],tf.int32))


        xs=prefer_static.concat([[-1],self._shape[:-1],[1,self._hparams.num_logistic_mix]],axis=0)
                
        

                         
        m2=self._locs[:,:,:,1,:]+coeffs_i[:,:,:,0,:]*x[:, :, :, 0, :]
                
                
        m2=tf.reshape(m2,xs)
                              
        m3=self._locs[:,:,:,2,:]+coeffs_i[:,:,:,1,:]*x[:, :, :, 0, :]+coeffs_i[:,:,:,2,:]*x[:, :, :, 1, :]
                
        m3=tf.reshape(m3,xs)

        self._locs = tf.concat([tf.reshape(self._locs[:,:,:,0,:],xs), m2, m3],3)


        self._log_scales=tf.maximum(self._log_scales,self._hparams.log_scale_low_bound)
                                   



    def sample(self,condition,training):
        """ 
            It draws samples from the mixture of the discretized logistic distributions.
            
            Args:
                condition: Tensor that contains the conditioning factor of the distribution.
                training:  Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                      
        
            Returns:
                The sample, a tensor of the shape: [batch_size,]+self._shape
        """
        
        with tf.name_scope(self.name or 'DiscretizedLogisticMixture_sample'):
            self.call(inputs=condition,training=training)
            
            
            params=self._params
            ## (batch_size,dim1,dim2,num_mix)
            logits_probs=params[:,:,:,:self._hparams.num_logistic_mix]
            
            
            ### sample categorical distribution
            
            # use the  gumbel trick: https://en.wikipedia.org/wiki/Categorical_distribution#Sampling_via_the_Gumbel_distribution
            # https://stats.stackexchange.com/questions/64081/how-do-i-sample-from-a-discrete-categorical-distribution-in-log-space
            
            sel = tf.one_hot(tf.math.argmax(logits_probs - tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(logits_probs), minval=1e-5, maxval=1. - 1e-5))), 3), depth=self._hparams.num_logistic_mix, dtype=tf.float32)
            
            ## add a channel dimension
            sel=tf.reshape(sel,shape=prefer_static.concat([[-1],self._shape[:-1],[1,self._hparams.num_logistic_mix]], axis=0))
            ## (batch_size,dim1,dim2,1, nr_mix)
            
            
            
            
            
            ## (batch_size,dim1,dim2,num_mix, 3-for the regression coeffs +2*3)
            params=tf.reshape(params[:,:,:,self._hparams.num_logistic_mix:],prefer_static.concat([[-1],self._shape[:-1],[self._hparams.num_logistic_mix,2*self._shape[-1]+3]], axis=0))
                
                
            ## (batch_size,image_dim_1,image_dim_2,num_mix,3)
            coeffs=tf.math.tanh(params[:,:,:,:,:3])
            ## (batch_size,image_dim_1,image_dim_2,3,num_mix)
            coeffs=tf.keras.backend.permute_dimensions(coeffs,pattern=[0,1,2,4,3])
                
            params=tf.reshape(params[:,:,:,:,3:],prefer_static.concat([[-1],self._shape,[self._hparams.num_logistic_mix*2]], axis=0))
            ## (batch_size,image_dim_1,image_dim_2,num_channels,num_mix)
            locs, log_scales = tf.split(params, 2, axis=-1)
            
            
            log_scales=tf.maximum(log_scales,self._hparams.log_scale_low_bound)
                 
                
            ## select loc from the mixture components
            
            locs=tf.reduce_sum(tf.math.multiply(locs,sel),4)
            
            ## select log_scales from the mixture components
            log_scales=tf.reduce_sum(tf.math.multiply(log_scales,sel),4)
            scales = tf.math.exp(log_scales)
                
            ## select coeffs from the mixture components

            coeffs=tf.reduce_sum(tf.math.multiply(coeffs,sel),4)
            
            
            ## https://stackoverflow.com/questions/3955877/generating-samples-from-the-logistic-distribution
                
            u = tf.random.uniform(tf.shape(locs), minval=1e-5, maxval=1. - 1e-5)
                
            x = tf.math.add(locs,tf.math.multiply(scales,(tf.math.log(u) - tf.math.log(1. - u))))
            
                
            x0 = tf.minimum(tf.maximum(x[:,:,:,0], -1.), 1.)
            x1 = tf.minimum(tf.maximum(x[:,:,:,1] + coeffs[:,:,:,0]*x0, -1.), 1.)
            x2 = tf.minimum(tf.maximum(x[:,:,:,2] + coeffs[:,:,:,1]*x0 + coeffs[:,:,:,2]*x1, -1.), 1.)
                
            xs=prefer_static.concat([[-1],self._shape[:-1],[1]],axis=0)
    
                
            x=tf.concat([tf.reshape(x0,xs), tf.reshape(x1,xs), tf.reshape(x2,xs)],3)

            ## from (-1,1) to (0,255)
            ## x=self._hparams.low + 0.5 * (self._hparams.high - self._hparams.low) * (x + 1.)
            
            ## from (-1,1) to (0,1), return it as a tf.float32
            x= 0.5  * (x + 1.)
            
            return x
            




    def log_prob(self,x,condition,training=True,crop_size=0):
        
        """
         It computes the log-likelihood of the samples.
            Args:
                x:         Tensor that contains the samples whose log-likleihood to be computed. It should have shape: [batch_size,]+self._shape
                condition: Tensor that contains the conditioning factor of the distribution. The batch dimension should match that of x.
                training:  Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                crop_size: Exclude marginal pixels (not used currently).  
            Returns:
                Tensor of shape [batch_size, ] containing the log probability.
        """


        with tf.name_scope(self.name or 'DiscretizedLogisticMixture_log_prob'):
            
            
            self.call(inputs=condition,training=training)

            
            self.create_params(x)

          
            shape_=  self._shape if tensorshape_util.rank(self._shape)>1 else  prefer_static.concat([self._shape,[1]],axis=0)

            x = tf.reshape(x, prefer_static.concat([[-1], shape_,[1]],axis=0)) + tf.zeros(prefer_static.concat([shape_,[self._hparams.num_logistic_mix]],axis=0))


    
        
        def compute_logp(logits_probs,locs,log_scales):
            """
                logits_probs: [batch_size,image_dim_1,image_dim_2,num_mixtures]
                locs,log_scales: [batch_size,image_dim_1,image_dim_2,channels,num_mixtures]
            """
            
 
        
            centered_x = x - locs

            inv_stdv = tf.exp(-log_scales)
            plus_in = inv_stdv * (centered_x + 1./255.)
            cdf_plus = tf.nn.sigmoid(plus_in)
            min_in = inv_stdv * (centered_x - 1./255.)
            cdf_min = tf.nn.sigmoid(min_in)
            log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
            log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
            cdf_delta = cdf_plus - cdf_min # probability for all other cases
            mid_in = inv_stdv * centered_x

        
            log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in)
            
            ## [B,image_dim1,image_dim2,channels,num of mixtures]: log probability for each pixel and according to each component of the mixture
            log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.99, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf1.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))
    
            ## for a single pixel (k,l) in [image_dim1,image_dim2], its log probability (across all its channels) is :
            ## LL=log Sum_i [\pi_i * P_i] = log Sum_i [\pi_i * exp(log(P_i))]: where i the i-th compnent of the mixture, Pi the probability of the pixel (across the channels) according to mixture i
            ## \pi_i is the contribution of each mixture. \pi_i=exp(log \pi_i)
            ## log(P_i) is tf.reduce_sum(log_probs,3), log \pi_i is self._log_prob_from_logits(logits_probs)
            ## LL= log Sum_i exp [log \pi_i + log(P_i)]
           
            log_probs = tf.reduce_sum(log_probs,3 if tensorshape_util.rank(self._shape)>1  else 2) + self._log_prob_from_logits(logits_probs)
            ## sum across all pixels in [image_dim1,image_dim2]
            logp=tf.reduce_sum(self._log_sum_exp(log_probs),axis=[1,2] if tensorshape_util.rank(self._shape)>1 else 1)
            return logp
                    
                  

        logp=compute_logp(self._logits_probs,self._locs,self._log_scales)


        return logp


    @staticmethod
    def _log_sum_exp(x):
        """ numerically stable log_sum_exp implementation that prevents overflow """
        axis = len(x.get_shape())-1
        m = tf.reduce_max(x, axis)
        m2 = tf.reduce_max(x, axis, keepdims=True)
        s= m + tf1.log(tf.reduce_sum(tf.exp(x-m2), axis))


        return s

    @staticmethod
    def _log_prob_from_logits(x):
        """ numerically stable log_softmax implementation that prevents overflow """


        ## logits: z1,z2,...,zk
        ## probs from logits: pi= exp(zi)/(exp(z1)+exp(z2)+...+exp(zk)) (log softmax)
        
        ## log probs log pi=zi -log(sum_i_to_k exp(zi)))
        ## log(sum_i exp (zi)) trick: compute m=max xi=xj
        ## log(sum_i exp (zi)) =m+log(exp(x1-m)+exp(x2-m)+....+exp(xk-m))
        axis = len(x.get_shape())-1
        m = tf.reduce_max(x, axis, keepdims=True)
        

        return x - m - tf1.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))



    @staticmethod
    def get_default_hparams():
        """ default parameters for the discretized logistic mixture  layer """
        return HParams(num_logistic_mix=5,            # number of mixture components
                       high=255,                      # highest possible discrete value
                       low=0,                         # lowest possible discrete value
                       log_scale_low_bound=-7.0,      # lower bound for the log-scale, to ease stability.
                       log_scale_upper_bound=7.0,     # upper bound for the log-scale, to ease stability.
                       )


