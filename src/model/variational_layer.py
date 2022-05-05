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


import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow_probability as tfp
import numpy as np

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import prefer_static
from stat_tools import DISTRIBUTION
from layers import LAYER, DepthWiseAttention
from layers.util import DeepVaeInit
import tensorflow_addons as tfa


__all__ = [
           'VariationalLayer',
           ]



class VariationalLayer(tfk.layers.Layer):
    
    """
        *** Variational Layer ****
        
        References:
            [1] Apostolopoulou, I., Char, I., Rosenfeld, E. and Dubrawski, A., 2022, April. 
                Deep Attentive Variational Inference. In International Conference on Learning Representations.
                
        Args:
            posterior_type:               Class of distribution for the posterior.
            prior_type:                   Class of distribution for the prior.
            latent_shape:                 Shape of the latent features.
            posterior_hparams:            Parameters of the posterior distribution.
            posterior_network_hparams:    Network hyper-parameters of the cell generating posterior parameters.
            prior_hparams:                Parameters for prior distribution.
            prior_network_hparams:        Network hyper-parameters of the cell generating prior parameters.
            encoder_hparams:              Hyper-parameters for the residual cell per layer of the encoder.
            decoder_hparams:              Hyper-parameters for the residual cell per layer of the decoder.
            merge_decoder_hparams:        Hyper-parameters for the cell that merges latent sample and stochastic context in the decoder.
     `      merge_encoder_hparams:        Hyper-parameters for the cell that merges the deterministic and stochastic context  in the encoder.
            constant_decoder:             Flag indicating whether the variational layer is the first in the hierarchy, thus it is not receiving stochastic context from previous layer.
            decoder_feature_shape:        In case constant_decoder=True, it indicates the shape of the constant tensor.
            residual:                     Flag indicating whether the residual parametrization between the prior and the posterior will be used.
            decoder_attention_hparams:    Depth-wise attention parameters in the decoder.
            encoder_attention_hparams:    Depth-wise attention parameters in the encoder.
            is_last_layer:                Flag indicating whether the variational layer is the last in the hierarchy,
    """
    
    
    def __init__(self,
                 posterior_type,
                 prior_type,
                 latent_shape,
                 posterior_hparams,
                 posterior_network_hparams,
                 prior_hparams=None,
                 prior_network_hparams=None,
                 encoder_hparams=None,
                 decoder_hparams=None,
                 merge_encoder_hparams=None,
                 merge_decoder_hparams=None,
                 constant_decoder=False,
                 decoder_feature_shape=None,
                 residual=False,
                 decoder_attention_hparams=None,
                 encoder_attention_hparams=None,
                 is_last_layer=False,
                 **kwargs):
        
        super(VariationalLayer, self).__init__(dtype=tf.float32,**kwargs)
        
        # set hyper-parameters
        
        self._latent_shape=latent_shape
        
        self._posterior_type=posterior_type
    
        self._prior_type=prior_type
        
        self._latent_shape=latent_shape
        
        self._residual=residual
        
        self._posterior_hparams=posterior_hparams
        
        self._posterior_network_hparams=posterior_network_hparams
    
        self._prior_hparams=prior_hparams
        
        self._prior_network_hparams=prior_network_hparams
        
        self._encoder_hparams=encoder_hparams
        
        self._decoder_hparams=decoder_hparams
        
        self._constant_decoder=constant_decoder
        
        self._merge_encoder_hparams=merge_encoder_hparams
        
        self._merge_decoder_hparams=merge_decoder_hparams
        
        self._decoder_attention_hparams=decoder_attention_hparams
        
        self._is_last_layer=is_last_layer
        
        
        self._decoder_feature_shape=decoder_feature_shape
        
        self._encoder_attention_hparams=encoder_attention_hparams
                
        # build residual cells for encoder
        if self._encoder_hparams is not None:
            self._encoder_network=LAYER[self._encoder_hparams.layer_type](hparams=self._encoder_hparams)
        else:
            self._encoder_network=tf.keras.layers.Lambda(lambda x:x)
            
        # build residual cells for decoder
        if self._decoder_hparams is not None:       
            
            if self._constant_decoder:
                self._decoder_network=tfp.layers.VariableLayer(shape=prefer_static.concat([[1],decoder_feature_shape], axis=0),
                                                                 initializer=DeepVaeInit(),
                                                                 regularizer=self._decoder_hparams.bias_regularizer,
                                                                 )
                
            else:
                self._decoder_network=LAYER[self._decoder_hparams.layer_type](hparams=self._decoder_hparams)
        else:
            self._decoder_network=tf.keras.layers.Lambda(lambda x:x)
        
        ## it concatenates the stochastic context and latent samples from previous layer
        if self._merge_decoder_hparams is not None:       
            
            self._merge_decoder_network=LAYER[self._merge_decoder_hparams.layer_type](hparams=self._merge_decoder_hparams,
                                                                                                num_filters=decoder_feature_shape[-1])
        else:
            self._merge_decoder_network=tf.keras.layers.Lambda(lambda x:tf.concat(x,axis=-1))
            
            
        ## it concatenates the stochastic and deterministic context  
        if self._merge_encoder_hparams is not None:    
            if self._encoder_attention_hparams is not None:
                merge_encoder_num_filters=self._encoder_hparams['num_filters']+self._encoder_attention_hparams.key_dim
            else:
                merge_encoder_num_filters=None
                
            ## it concatenates the stochastic feature of the previous generative layer with the deterministic features of the encoder of the current layer            
            self._merge_encoder_network=LAYER[self._merge_encoder_hparams.layer_type](hparams=self._merge_encoder_hparams,
                                                                                            num_filters=merge_encoder_num_filters)
        else:
            self._merge_encoder_network=tf.keras.layers.Lambda(lambda x:tf.concat(x,axis=-1))
            
        # build the posterior distribution
        self._posterior_network=DISTRIBUTION[self._posterior_type](shape=self._latent_shape,
                                                                   hparams=self._posterior_hparams,
                                                                   network_hparams=self._posterior_network_hparams,
                                                                   )
                                                                            
        
        # build the prior distribution
        if self._prior_hparams is not None:
            self._prior_network=DISTRIBUTION[self._prior_type](shape=self._latent_shape,
                                                               hparams=self._prior_hparams,
                                                               network_hparams=self._prior_network_hparams,
                                                               )
        else:
            self._prior_network=None
            
            
        ## depth-wise, top-down attention in decoder
        self._use_decoder_attention=False  
        if self._decoder_attention_hparams is not None:
            
            self._use_decoder_attention=True 
            self._decoder_attention= DepthWiseAttention(hparams=self._decoder_hparams,
                                                 attention_hparams=self._decoder_attention_hparams
                                                 )
             
            #normalization scheme
            if self._decoder_attention_hparams.use_layer_norm:
                self._attention_layer_norm_layer=tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                                gamma_regularizer=self._decoder_hparams.batch_norm_regularizer,
                                                                                beta_regularizer=self._decoder_hparams.batch_norm_regularizer)
             
                self._attention_layer_norm_layer2=tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                                gamma_regularizer=self._decoder_hparams.batch_norm_regularizer,
                                                                                beta_regularizer=self._decoder_hparams.batch_norm_regularizer)
             
                self._ctx_layer_norm_layer=tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                          gamma_regularizer=self._decoder_hparams.batch_norm_regularizer,
                                                                          beta_regularizer=self._decoder_hparams.batch_norm_regularizer)
            # scale for residual connection            
            self._gamma=self.add_weight(name='gamma',
                                        shape=(),
                                        initializer=tf.zeros_initializer(),
                                        regularizer=self._decoder_hparams.bias_regularizer,
                                        trainable=True)
       
            
        ## depth-wise, bottom-up attention in encoder
        self._use_encoder_attention=False  
        if self._encoder_attention_hparams is not None:
            self._use_encoder_attention=True
            self._encoder_attention=DepthWiseAttention(hparams=self._encoder_hparams,
                                              attention_hparams=self._encoder_attention_hparams,
                                              )
            # scale for residual connection    
            self._encoder_gamma=self.add_weight(name='enc_gamma',
                                                shape=(),
                                                initializer=tf.zeros_initializer(),
                                                regularizer=self._encoder_hparams.bias_regularizer,
                                                trainable=True)
            
            #normalization scheme
            if self._encoder_attention_hparams.use_layer_norm:
             
                self._cond_attention_layer_norm_layer=tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                                      gamma_regularizer=self._encoder_hparams.batch_norm_regularizer,
                                                                                      beta_regularizer=self._encoder_hparams.batch_norm_regularizer)
                
             
                self._encoder_attention_layer_norm_layer=tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                                        gamma_regularizer=self._encoder_hparams.batch_norm_regularizer,
                                                                                        beta_regularizer=self._encoder_hparams.batch_norm_regularizer)
                 

            
            
    def compute_context(self, latent_condition=None, context=None,training=False,num_samples=1):
        
        """
           It computes stochastic context of layer.
           
           Args:
               latent_condition: 4D tensor of the latent sample.
               context:          4D tensor of stochastic context.
               training: Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
               num_samples: Batch dimension of stochastic context. Used only for the first layer of the hierarchy.
            Return:
                4D tensor of new stochastic context.
        """

        if latent_condition is not None:
            latent_condition = tf.convert_to_tensor(latent_condition, dtype=self.dtype, name='previous condition')
            if context is not None:
                    
                context = tf.convert_to_tensor(context, dtype=self.dtype, name='context')
                # merge latent sample and stochastic context
                prev_context=self._merge_decoder_network([context,latent_condition])#
            else:
                prev_context=latent_condition

                
            # call residual cell of decoder
            return self._decoder_network(prev_context,training=training)
            
        # a constant context is used
        if self._constant_decoder:
            context_condition=self._decoder_network(0.0)
            return tf.tile(context_condition,multiples=[num_samples,1,1,1])

        return None
                
                
    def generate(self, latent_condition=None, context=None, num_samples=1):
        """
            It samples latent factors from the prior distribution.
            
            Args:
               latent_condition: 4D tensor of the latent sample.
               context:          4D tensor of stochastic context.
               num_samples: Batch dimension of stochastic context. Used only for the first layer of the hierarchy.
            Return:
                xi: 4D tensor of latent sample.
                context: 4D tensor of stochastic context.
        """
        #create context of layer
        new_context,prior_condition,key,_ =self.call_decoder(False,  latent_condition, context, num_samples)
    
        # call prior network, if any
        if self._prior_network is not None:
            xi=self._prior_network.sample(condition=prior_condition,training=False,num_samples=num_samples)
      
        else:
            latent_shape=self._latent_shape
            loc=np.zeros(latent_shape , dtype=np.float32)
            scale_diag=np.ones(latent_shape , dtype=np.float32)
            prior = tfp.distributions.MultivariateNormalDiag(loc=loc,scale_diag=scale_diag)
            xi=prior.sample(sample_shape=num_samples)


        xi=tf.reshape(xi,shape=prefer_static.concat([[-1], self._latent_shape], axis=0))
        
        # return sample, context and keys, if needed
        if not self._use_decoder_attention:
            return xi, prior_condition
        if not self._is_last_layer:
            return xi, tf.concat([new_context,key],axis=-1)
        
        return xi, new_context

        
        
        
    def call_decoder(self, training,  latent_condition=None, context=None, num_samples=1):
        """
            It calls the decoder of layer.
            
            Args:
               latent_condition: 4D tensor of the latent sample.
               context:          4D tensor of stochastic context.
               num_samples: Batch dimension of stochastic context. Used only for the first layer of the hierarchy.
               
            Return:
               
                new_context: 4D tensor of final stochastic context.
                prior_condition: 4D tensor for the condition of prior.
                key: 4D tensor for key of stochastic context (if needed).
                enc_query: 4D tensor for query to encoder (if needed).
        """
        with tf.name_scope(self.name or 'VariationalLayer_decoder'):
            
            # split context to values and keys
            if self._use_decoder_attention:
                gen_num_filters=context[:,-1,:,:,:].shape[-1]-self._decoder_attention_hparams.query_dim
                values,keys=tf.split(context,[gen_num_filters,self._decoder_attention_hparams.query_dim],axis=-1)
                context=values[:,-1,:,:,:]
     
            
            # merge and process input from previous layer
            new_context= self.compute_context(latent_condition=latent_condition, context=context,training=training,num_samples=num_samples)
            key=None
            
            # extract keys and queries from context, if needed
            if self._use_decoder_attention:
                
                query_dim=self._decoder_attention_hparams.query_dim
                
                key_dim=self._decoder_attention_hparams.key_dim
                
     
                if not self._is_last_layer:
                    gen_num_filters=new_context.shape[-1]-key_dim-query_dim
                    new_context,key,query=tf.split(new_context,[gen_num_filters,key_dim,query_dim],axis=-1)
                else:
                    gen_num_filters=new_context.shape[-1]-query_dim
                    new_context,query=tf.split(new_context,[gen_num_filters,query_dim],axis=-1)#  
           
                
            # extract query to encoder from context, if needed     
            enc_query=None   
            if self._use_encoder_attention:
                enc_query=new_context[:,:,:,:self._encoder_attention_hparams.query_dim]

                
            # apply depthwise attention in encoder   
            if self._use_decoder_attention:
                
                # normalize context
                if self._decoder_attention_hparams.use_layer_norm:
                    new_context=new_context+tfa.activations.gelu(self._ctx_layer_norm_layer(new_context))

                values=tf.unstack(values,axis=1)
                
                if self._decoder_attention_hparams.use_layer_norm:
                    values[-1]=self._attention_layer_norm_layer(values[-1])
                
                values=tf.stack(values,axis=1)
                
                values=self._decoder_attention.apply(query=query,
                                                       keys=keys,
                                                       values=values
                                                       )
                # normalize attentionr result
                if self._decoder_attention_hparams.use_layer_norm:
                    values=values+tfa.activations.gelu(self._attention_layer_norm_layer2(values))
              
                # residual connection
                prior_condition=new_context+self._gamma*values
                
            else:
                
                prior_condition=new_context
                
            return new_context,prior_condition,key,enc_query
                
           
                
    def infer(self,condition,direction,training,latent_condition=None, context=None,num_samples=1,compute_q_logp=False,compute_p_logp=False):
        """
            Inference method.
            
            Inputs:
                condition:         4D tensor of deterministic context.
                direction:         Mode: bottom-up vs top-down inference.
                latent_condition: 4D tensor of the latent sample.
                context:          4D tensor of stochastic context.
                num_samples:      Batch dimension of stochastic context. Used only for the first layer of the hierarchy.
                comput_q_logp:    Flag indicating whether the likelihood of the posterior distribution for the samples drawn during the inference to be returned.
                comput_p_logp:    Flag indicating whether the likelihood of the prior distribution for the samples drawn during the inference to be returned.
            
            Returns:
                xi:                4D tensor of new latent sample.
                new_context:       4D tensor of new stochastic context.
                kl_loss:           kl divergence between posterior and the prior.
                q_logp:            Posterior log-ikelihood if comput_q_logp=True, else 'None'.
                p_logp:            Prior log-likelihood if comput_p_logp=True, else 'None'.
        """
        
        with tf.name_scope(self.name or 'VariationalLayer_infer'):
            
            ## process evidence condition of current layer
            condition = tf.convert_to_tensor(condition, dtype=self.dtype, name='condition')
            
            ## bottom-up pass
            if direction=='bottom-up':
                    
                return self._encoder_network(condition,training)
            
            new_context,prior_condition,key,enc_query =self.call_decoder(training,  latent_condition, context, tf.shape(condition)[0])
                
            prior_params=None
            
            not_fist_layer=latent_condition is not None
            
            ## for the prior distribution
            if not_fist_layer:
                
                # # form the prior distribution
                self._prior_network.call(condition=prior_condition,training=training)
                    
                # extract the prior parameters
                if self._residual:
                    prior_params=self._prior_network.params()
                
            ## extract the keys and the queries
            if self._use_encoder_attention:

                values,keys=tf.split(condition,[self._encoder_hparams['num_filters'],self._encoder_attention_hparams.key_dim],axis=-1)
                condition=values[:,0,:,:,:]
                        
            ## merge the stochastic and the deterministic context for the posterior           
            if not_fist_layer:
                condition=self._merge_encoder_network([condition,prior_condition])
                
            ## attend to get the final context of the posterior
            if self._use_encoder_attention:   
                
                if not_fist_layer:
                    condition,new_key=tf.split(condition,[self._encoder_hparams['num_filters'],self._encoder_attention_hparams.key_dim],axis=-1)
                    
                    ## replace the last value (which has only evidence), with the one after merging latent and evidence
                    values=tf.unstack(values,axis=1)
                    values[0]=condition
                    values=tf.stack(values,axis=1)
                    
                    ## replace the last key (which has only evidence), with the one after merging latent and evidence
                    ## it will be used as key for attention of this and the next layers in the hierarchy
                    keys=tf.unstack(keys,axis=1)
                    keys[0]=new_key
                    keys=tf.stack(keys,axis=1)
                    
                values=self._encoder_attention.apply(query=enc_query,
                                                     keys=keys,
                                                     values=values
                                                     )
                
                if self._encoder_attention_hparams.use_layer_norm:
                    values=values+tfa.activations.gelu(self._encoder_attention_layer_norm_layer(values))
                    condition=condition+tfa.activations.gelu(self._cond_attention_layer_norm_layer(condition))
                    
                condition=condition+self._encoder_gamma*values
            
                condition=tf.concat([condition,keys[:,0,:,:,:]],axis=-1)        
                        
            
            # sample posterior and compute log-probability of the sample
            xi=self._posterior_network.sample(condition= condition,training=training,num_samples=num_samples,initial_params=prior_params)
           
            q_logp=None#tf.zeros(shape=(num_samples,tf.shape(condition)[0]))
            if compute_q_logp:
                q_logp=self._posterior_network.log_prob(x=xi, condition= condition,training=training,initial_params=prior_params,compute_dist=False)
            
                
            # compute prior log-probability of the sample
            p_logp=None
            if compute_p_logp:
                if self._prior_network is not None:
                    p=self._prior_network._p[-1]
                else:
                    p = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(self._latent_shape, dtype=np.float32),scale_diag=np.ones(self._latent_shape, dtype=np.float32))
                    
                p_logp=p.log_prob(xi)
                p_logp=tf.reduce_sum(p_logp,axis=[-1-i for i in range(tensorshape_util.rank( self._latent_shape)-1)])
   
            
            xi=tf.reshape(xi,shape=prefer_static.concat([[-1], self._latent_shape], axis=0))
   
            ## compute kl (posterior||prior)
            if self._prior_network is not None:
                p=self._prior_network._p[0]
            else:
                # if there is no prior network, the standard normal is considered
                p = tfp.distributions.MultivariateNormalDiag(loc=np.zeros(self._latent_shape, dtype=np.float32),scale_diag=np.ones(self._latent_shape, dtype=np.float32))
                
            q=self._posterior_network._p[-1]
            kl_loss=tfp.distributions.kl_divergence(q,p)

            kl_loss=tf.reduce_sum(kl_loss,axis=[i for i in range(1,tensorshape_util.rank(self._latent_shape))])
            
            
            if not self._use_decoder_attention:
                return xi, prior_condition, kl_loss, q_logp, p_logp
            if not self._is_last_layer:
                return xi, tf.concat([new_context,key],axis=-1), kl_loss, q_logp, p_logp
        
            return xi, new_context, kl_loss, q_logp, p_logp
        


                       

