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
from tensorflow_probability.python.internal import tensorshape_util


import numpy as np
import copy

from layers import LAYER
from model.variational_layer import VariationalLayer
from model.data_layer import DataLayer
from model.util import _DeepVAE



__all__ = ['DeepVAE']



class DeepVAE(tf.keras.Model,_DeepVAE):
    """  
        *** Deep Attentive Variational AutoEncoder ****
         
         References:
             [1] Apostolopoulou, I., Char, I., Rosenfeld, E. and Dubrawski, A., 2022, April. 
                 Deep Attentive Variational Inference. In International Conference on Learning Representations.
         Args:  
            name: Python `str` name prefixed to Ops created by this class. Default: subclass name.
            List of model hyperparameters. See _DeepVAE in model/util.py for details.
         
    """
    def __init__(self,
                 name='deep_autoencoder',
                 **kwargs
                 ):

        # set the hyperparaneters
        _DeepVAE.__init__(self, **kwargs)
    
        # initialize tensorflow model
        tf.keras.Model.__init__(self,name, dtype=tf.float32)
        
        ###############################################################################################
        #                                  Build Data Distribution                                    #
        ###############################################################################################   
        
        if self._hparams.num_layers >1 and self._merge_decoder_hparams is not None:
            self._merge_decoder=LAYER[self._merge_decoder_hparams.layer_type](hparams=self._merge_decoder_hparams)
        else:
            self._merge_decoder=tf.keras.layers.Lambda(lambda x:x)
                
        self._data_layer=DataLayer(data_dist_hparams=self._data_dist_hparams,
                                   network_hparams=self._dist_network_hparams.data,
                                   preproc_hparams=self._postproc_decoder_hparams,
                                   class_dist=self._hparams.data_distribution,
                                   image_info=self._image_info,
                                   num_proc_blocks=self._num_proc_blocks,
                                   )



        
        
        downscale_factor=2**self._num_proc_blocks
        
        ## number of features in the recognition network for the current scale
        num_encoder_filters=self._preproc_encoder_hparams['num_filters']
     
        ###############################################################################################
        #                               Build Data Preprocessing Layers                               #
        ###############################################################################################
        
        self._preproc_encoder=[]
        for s in range(self._num_proc_blocks):
            preproc_encoder_hparams=copy.deepcopy(self._preproc_encoder_hparams)
           
            if self._encoder_attention_hparams is not None:
                preproc_encoder_hparams['num_filters']=num_encoder_filters+int(self._encoder_attention_hparams.key_dim/downscale_factor)
            else:
             
                preproc_encoder_hparams['num_filters']=num_encoder_filters
                

            if self._preproc_encoder_hparams is not None:
                self._preproc_encoder.append(LAYER[self._preproc_encoder_hparams.layer_type](hparams=preproc_encoder_hparams))
            else:
                self._preproc_encoder.append(tf.keras.layers.Lambda(lambda x:x))
                
            if s+1<self._num_proc_blocks:
                preproc_encoder_hparams['use_stem']=False
            num_encoder_filters=num_encoder_filters*2

            
        ###############################################################################################
        #                             Build Variational Hierarchy                                     #
        ###############################################################################################
        
        # hierarchy of inference network
        self._latent_layer =[] 
            
        if self._decoder_attention_hparams is not None:
            self._ctx_layer_norm=[]
             
        image_shape=self._image_info['shape'][0]+self._image_info['pad_size']*2


        
        latent_width=int(image_shape/downscale_factor)
        latent_height=int(image_shape/downscale_factor)
        
        
        ## latent shape per layer
        self._layer_latent_shape=[]
        

        ## number of features in the generative network for the current scale
        num_decoder_filters=self._decoder_hparams['num_filters']*downscale_factor
            
        generative_feature_shape=[latent_width,latent_height,num_decoder_filters]
        
        layer_latent_shape=tf.TensorShape([latent_width,latent_height,self._hparams.layer_latent_shape[-1]])
             
        encoder_hparams=copy.deepcopy(self._encoder_hparams)
        encoder_hparams['num_filters']=num_encoder_filters
          
            
        for i in range(self._num_layers):
                
                
            self._layer_latent_shape.append(layer_latent_shape)
                
            generative_feature_shape_i=copy.deepcopy(generative_feature_shape)
                
            # depth-wise attention in the decoder
            if self._decoder_attention_hparams is not None:
           
                if i>0:
                    ## the layer should generate its query
                    query_dim=self._decoder_attention_hparams.query_dim
                    generative_feature_shape_i[-1]=generative_feature_shape_i[-1]+query_dim
                
                if i<self._num_layers-1:
                    key_dim=self._decoder_attention_hparams.key_dim
                    ## the layer should generate its key for the next layers in the hierarchy
                    generative_feature_shape_i[-1]=generative_feature_shape_i[-1]+key_dim
                    
            # depth-wise attention in the encoder
            if self._encoder_attention_hparams is not None:
                query_dim=self._encoder_attention_hparams.query_dim
                generative_feature_shape_i[-1]+=query_dim
                
            #create variational layer
            self._latent_layer.append( VariationalLayer(posterior_type=self._hparams.posterior_type,
                                                             prior_type=self._hparams.prior_type,
                                                             latent_shape=layer_latent_shape,
                                                             encoder_hparams=encoder_hparams,
                                                             decoder_hparams=self._decoder_hparams if (i>0 or  self._num_layers >1 ) else None,
                                                             merge_decoder_hparams=self._merge_decoder_hparams if i>0 else None,
                                                             merge_encoder_hparams=self._merge_encoder_hparams if i>0 else None,
                                                             posterior_hparams=self._posterior_hparams,
                                                             posterior_network_hparams=self._dist_network_hparams.posterior if i>0 else self._dist_network_hparams.init_posterior,
                                                             prior_hparams=self._prior_hparams if i>0 else None,
                                                             prior_network_hparams=self._dist_network_hparams.prior if i>0  else None,
                                                             residual=self._hparams.residual_var_layer,
                                                              constant_decoder=(i==0 and self._num_layers >1 ),
                                                             decoder_feature_shape=tf.TensorShape(generative_feature_shape_i),           
                                                             decoder_attention_hparams = self._decoder_attention_hparams if i>0 else None,
                                                             is_last_layer=(i==self._num_layers-1),
                                                             encoder_attention_hparams = self._encoder_attention_hparams,
                                                             ))
            
            # normalization layer for the depth-wise attention in decoder
            if self._decoder_attention_hparams is not None:
                self._ctx_layer_norm.append(tf.keras.layers.LayerNormalization(epsilon=1e-5,
                                                                           gamma_regularizer=self._decoder_hparams.batch_norm_regularizer,
                                                                           beta_regularizer=self._decoder_hparams.batch_norm_regularizer))
                    

                    


            self.built=False

        
    def build(self, input_shape):
                                                      
        super(DeepVAE, self).build(input_shape)
        
        self.count_params()
        
        
         
        
        
    def call(self,inputs,training=True):
        
        """
            It computes the NEBLO objective.
            Args:    
                inputs: 4D tensor with shape [batch_size,width, height, channels] with the images.
                training: Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
            
            Returns:
                nll: Conditional likelihood of the data.
                sum(kl): Total kl divergence of the variational model.
                nelbo: NELBO objective.
        """
    
        with tf.name_scope( 'deep_vae'):
            
            inputs=tf.keras.backend.cast(tf.convert_to_tensor(inputs, dtype=self.dtype),dtype='float32') 
            
            encoder_evidence=self.preprocess(inputs)
            
            # infer            
            c,h,kl,_,_=self.encode(inputs=encoder_evidence,training=training,compute_q_logp=False,compute_p_logp=False)
            
            # merge sample and stochastic context of last layer, if any
            if self._num_layers >1:
                cc=self._merge_decoder([c[-1],h[-1]])
            else:
                cc=h[0]
         
            # compute conditional, negative, data-likelihood
            if self._image_info['binary']:
                nll=self._data_layer.decode(latent_codes=cc,x=inputs,training=training)
            else:
                nll=self._data_layer.decode(latent_codes=cc,x=encoder_evidence,training=training)
                    
            
            nelbo=nll+sum(kl)
            

            return [nll,sum(kl),nelbo]




    def preprocess(self,x):
        """
            Utility function that processes the raw image pixels.
        """
        
        ## scale evidence to [-1,1]
        if not self._image_info['binary']:
            return  (2. * (x - self._image_info['low']) / (self._image_info['high'] - self._image_info['low'])) - 1.
        return  (2. * x) - 1.
        

   
    def encode(self,inputs,training,num_samples=1,compute_q_logp=False,compute_p_logp=False):
        
        """
            Bi-directional inference.
              
            Args: 
                inputs: 4D tensor with shape [batch_size,width,height,channel] with the processed image data.
                training: Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
                num_samples: Number of latent variable samples for inference.
                comput_q_logp: Flag indicating whether the likelihood of the posterior distribution for the samples drawn during the inference to be returned.
                comput_p_logp: Flag indicating whether the likelihood of the prior distribution for the samples drawn during the inference to be returned.
            Returns:
                c: Contexts of the decoder layers during the top-down pass of the model.
                h: Latent factors of the variational layer.
                kl: KL divergence for all the variational layers in the hierarchy.
                q_logp: Posterior log-ikelihood if comput_q_logp=True.
                p_logp: Prior log-likelihood if comput_p_logp=True.
        """

        with tf.name_scope( 'DeepVAE_encode'):
    
            inputs=tf.keras.backend.cast(tf.convert_to_tensor(inputs, dtype=self.dtype),dtype='float32')
            
            inputs=tf.tile(inputs,multiples=[num_samples,1,1,1])
            
            e=[None]*(self._num_layers+1)
            
            # encoder, pre-processing
            for s in range(self._num_proc_blocks):   
                          
                e[-1]=self._preproc_encoder[s](inputs if s==0 else e[-1],training)
            
            ## bottom-up pass
            for i in range(self._num_layers):
               
                j=self._num_layers-1-i    
                e[j]=self._latent_layer[j].infer(condition=e[j+1],  
                                                 direction='bottom-up',
                                                 training=training,)
               
            
        
            
            h=[None]*self._num_layers #the latent factors
            
            c=[None]*self._num_layers #context of the generative layer
            
            kl = [None]* self._num_layers #kl divergence for each layer
            
            q_logp = [None]*self._num_layers #posterior log-likelihood of the latent factors
            
            p_logp = [None]*self._num_layers #prior log-likelihood of the latent factors
            
            #top-down pass
            for i in range(self._num_layers):
                 
                #use all lower deterministic contexts, if attention is applied on the inference path
                if self._encoder_attention_hparams is not None:
                    condition=tf.stack(e[i:],axis=1)
                else:
                    condition=e[i]
                    
                if i>0:
                    
                    if self._decoder_attention_hparams is not None:                        
                        if i>1:
                            c[i-2]=self._ctx_layer_norm[i](c[i-2])
                        #use all upper stochastic contexts, if attention is applied on the decoder path
                        context=tf.stack(c[:i],axis=1)
                    else:
                        context=c[i-1]
                else:
                    # first layer, doesn't receive any context
                    context=None
                

                h[i],c[i],kl[i],q_logp[i],p_logp[i]=self._latent_layer[i].infer(condition = condition,
                                                                                latent_condition=h[i-1],
                                                                                context=context,
                                                                                direction='top-down',
                                                                                training=training,
                                                                                num_samples=num_samples,
                                                                                compute_q_logp=compute_q_logp,
                                                                                compute_p_logp=compute_p_logp)
                
                                
            if self._num_layers==1:
                return h,h,kl,q_logp,p_logp

            return c,h,kl,q_logp,p_logp
        

    def reconstruct(self,x):
        """
           Reconstruct the images for its latent representations.
           
           Args: 
               x, 4D tensor with shape [batch_size,width,height,channels] with the raw images to be reconstructed.
           Returns: 
               y, 4D tensor with shape [batch_size,width,height,channels] with the reconstructed image in the mapped space.
        """
        
        x=tf.keras.backend.cast(tf.convert_to_tensor(x, dtype=self.dtype),dtype='float32') 
        
        x=self.preprocess(x)
  
         
        # encode image
        c,h,_,_,_=self.encode(inputs=x,training=False,compute_q_logp=False,compute_p_logp=False)
 
        if self._num_layers >1:
            cc=self._merge_decoder([c[-1],h[-1]],training=False)
        else:
            cc=h[0]
        
        # decode image
        y=self._data_layer.generate(latent_codes=cc)
        
        return y
        
        
    def generate(self,num_samples=1):
        """
            Generates novel images from the model.
            
            Args:
                num_samples: number of new images to be drawn from the model.
            Returns:
                 h: the latent representation of y.
                y: the new sample images.
        """
  
            
        h=[None]*self._num_layers   # latent variables
            
        c=[None]*self._num_layers   # latent context
            
        # generate latent samples
        for i in range(self._num_layers):
            
            if i>0:
                    if self._decoder_attention_hparams is not None:                        
                        if i>1:
                            c[i-2]=self._ctx_layer_norm[i](c[i-2])
                            
                        context=tf.stack(c[:i],axis=1)
                    else:
                        context=c[i-1]
            else:
                    context=None
                
            h[i],c[i]=self._latent_layer[i].generate(latent_condition=h[i-1] if i>0 else None,
                                                     context=context,
                                                     num_samples=num_samples if (i==0 or self._prior_hparams is None) else 1,
                                                     )
            
    
        # merge latent sample and stochastic context of last later
        if self._num_layers >1:
            cc=self._merge_decoder([c[-1],h[-1]],training=False)
        else:
            cc=h[0]
        
        # generate images from the conditional. data distribution
        y=self._data_layer.generate(latent_codes=cc)
        

    
        return h,y
    
    def infer_worker_body(self,inputs,num_samples):

        encoder_inputs=self.preprocess(inputs)
                
        inputs_shape=inputs.shape[1:]
        
        inputs=tf.tile(inputs,[num_samples]+[1]*tensorshape_util.rank(inputs_shape))
        
        encoder_inputs=tf.tile(encoder_inputs,[num_samples]+[1]*tensorshape_util.rank(inputs_shape))
        
        c,h,_,q_logp,p_logp=self.encode(inputs=encoder_inputs,training=False,num_samples=1,compute_q_logp=True,compute_p_logp=True)

        
        l=tf.math.add_n([ tf.math.subtract(p_logp[j],q_logp[j]) for j in range(self._num_layers)])

        if self._num_layers >1:
            cc=self._merge_decoder([c[-1],h[-1]],training=False)
        else:
            cc=h[0]
         
        if self._image_info['binary']:
            nll=self._data_layer.decode(latent_codes=cc,x=inputs,training=False)
        else:
            nll=self._data_layer.decode(latent_codes=cc,x=encoder_inputs,training=False)
            
      
        l=tf.math.add(l,-nll)
         
        l=tf.reshape(l,shape=(num_samples,-1))

        return tf.math.reduce_logsumexp(l,axis=0)
    

    def predict(self, x, num_samples=10,workers=10, *args, **kwargs):
        """
            Overwrites tensorflow implementation for customized model evaluation.
            Args:    
                x: 4D tensor with shape [batch_size,width, height, channels] with the images.
                num_samples: Number of importance samples for estimation of the marginal likelihood.
                workers: Split to workers for memory efficiency.
        """
        #setting the evaluation parameters
        self.eval_samples= num_samples
        self.eval_workers= workers
        self.predict_function = None
        return super().predict(x, *args, **kwargs)




    def predict_step(self,inputs):
        """
            It computes an importance sampled estimate of the marginal likelihood of the model, as defined in the paper.
            References:
            [1]. Burda, Y., Grosse, R. and Salakhutdinov, R., 2015. Importance weighted autoencoders. arXiv preprint arXiv:1509.00519.
        
            Args:    
                inputs: 4D tensor with shape [batch_size,width, height, channels] with the images.
        """
        
        num_samples=self.eval_samples
        workers=self.eval_workers
                        
        logp=tf.zeros([workers,tf.shape(inputs)[0]])
        

        
        i = tf.constant(0)
        while_condition = lambda i,p: i<workers
        
        def call_worker(i,logp):
            
            num_samples_worker=tf.cond(tf.math.not_equal(i,workers-1), lambda: num_samples//workers, lambda: num_samples-(num_samples//workers)*(workers-1))

            logpi=self.infer_worker_body(inputs,num_samples_worker)
            
            
            logp = tf.cond(tf.math.equal(i,0), lambda:tf.expand_dims(logpi,0), lambda: tf.concat([logp, tf.expand_dims(logpi,0)], 0))
            
         

            return [i+1,logp]

        _, logp=tf.while_loop(while_condition, call_worker, [i,logp], shape_invariants=[i.get_shape(), tf.TensorShape([None,None])],parallel_iterations=1)
        

            
        return tf.math.subtract(tf.math.reduce_logsumexp(logp,axis=0),tf.math.log(tf.keras.backend.cast_to_floatx(tf.constant(num_samples))))
        
        
    
    def count_params(self,verbose=True):
        """ 
            Utility function that counts the number of the trainable parameters of the model.
            Args:
                verbose: Flag indicating verbose.
        """
        
  
        preproc_params=0
        for s in range(self._num_proc_blocks):
            preproc_params=preproc_params+int(np.sum([tfk.backend.count_params(p) for p in self._preproc_encoder[s].trainable_weights]))

        merge_gen_params=int(np.sum([tfk.backend.count_params(p) for p in self._merge_decoder.trainable_weights]))
        
        total_params=preproc_params+merge_gen_params
        for i in range(self._num_layers):
            var_params=int(np.sum([tfk.backend.count_params(p) for p in self._latent_layer[i].trainable_weights]))
            total_params+=var_params
            
        data_params=int(np.sum([tfk.backend.count_params(p) for p in self._data_layer.trainable_weights]))
        
        total_params+=data_params
        
        if verbose:
            print('Total Params:')
            print(total_params/1e6)
        
        return(total_params)







