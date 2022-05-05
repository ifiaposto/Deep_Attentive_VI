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

import copy
import tensorflow as tf
from util.hparams import HParams
from layers import LAYER


__all__ = ['_DeepVAE',            
           ]



class _DeepVAE():
    """  
        *** Deep Attentive Variational Inference **** 
        

        Args:
            hparams:                  Hyper-parameters of the variational architecture. See get_default_hparams() for a detailed description.
            preproc_encoder_hparams:  Hyper-parameters for the pre-processing layer in the encoder.
            
            encoder_hparams:          Hyper-parameters for the depthwise attention in the encoder.
            decoder_hparams:          Hyper-parameters for the depth-wise attention in the decoder.
            merge_decoder_hparams:    Hyper-parameters for the cell that merges the latent sample and the context in the decoder.
            merge_encoder_hparams:    Hyper-parameters for the cell that merges the deterministic and stochastic context in the encoder.
            postproc_decoder_hparams: Hyper-parameters for the post-processing layer in the decoder.
            data_dist_hparams:        Hyper-parameters for the data distribution.
            posterior_hparams:        Hyper-parameters for the posterior distribution.      
            prior_hparams:            Hyper-parameters for the prior distribution.
            image_info:               Dicitionary with info about the image to be generated.
                                        shape:Rank-3 shape of the image data to be generated.
                                        low: float number indicating the smallest possible value of the data.
                                        high: float number indicating the largest possible value of the data.
                                        pad_size:: integer number indicating the number of pixels to be croped from the generated image, otherwise non-positive.
            decoder_attention_hparams: Hyper-parameters for the depth-wise attention in the decoder.
            encoder_attention_hparams: Hyper-parameters for the depthwise attention in the encoder.
    """
    def __init__(self,
                hparams,
                preproc_encoder_hparams=None,
                encoder_hparams=None,
                decoder_hparams=None,
                merge_decoder_hparams=None,
                merge_encoder_hparams=None,
                postproc_decoder_hparams=None,
                data_dist_hparams=None,
                posterior_hparams=None,
                prior_hparams=None,
                image_info=None,
                decoder_attention_hparams=None,
                encoder_attention_hparams=None,
                 **kwargs,
                 ):
        
        
        super(_DeepVAE, self).__init__(**kwargs)
        
        ###############################################################################################
        #                                   Set HyperParameters                                       #
        ###############################################################################################
    
        
        ## shape of the data distribution
        self._image_info=image_info
        
        ## number of layers in the hierarchy 
        self._num_layers=hparams.num_layers
        
        ## number of pre-processing blocks in the inference network and post-processing blocks in the generative network.
        self._num_proc_blocks=hparams.num_proc_blocks
      
      
        self._hparams=hparams

        
        self._posterior_hparams=posterior_hparams
                

        self._data_dist_hparams=data_dist_hparams
        
        self._dist_network_hparams=_DeepVAE.get_default_network_hparams(self._hparams)

    
        self._preproc_encoder_hparams=preproc_encoder_hparams
        
        self._postproc_decoder_hparams=postproc_decoder_hparams
        
        self._encoder_hparams=encoder_hparams
        
        
        self._decoder_hparams=decoder_hparams
        
        self._merge_decoder_hparams=merge_decoder_hparams
        
        self._merge_encoder_hparams=merge_encoder_hparams

        # prior network hyperparameters
        # standard normal prior is used if this hparameter is None
        self._prior_hparams=prior_hparams
          
        self._decoder_attention_hparams=decoder_attention_hparams
        
        self._encoder_attention_hparams=encoder_attention_hparams
        
        self.eval_samples =1
        
        self.eval_workers =1
        
    @property
    def image_shape(self):
        ## exception if it's not positive
        
        if self._image_info['pad_size']>0:
            ps=self._image_info['pad_size']
            
            return [self._image_info['shape'][0]+2*ps, self._image_info['shape'][1]+2*ps,self._image_info['shape'][-1]]
        return self._image_info['shape']
    
    
    @image_shape.setter
    def image_shape(self, x):
        
        ## exception if it's not positive
        self._image_info['shape']=x
        
   
    @staticmethod
    def get_default_hparams():
        return HParams(posterior_type="gaussian_diag",                           #  class of distribution for the posterior.
                       prior_type="gaussian_diag",                               #  class of distribution for the prior.
                       data_distribution="discretized_logistic_mixture",         #  class of distribution for the data.
                       num_layers=1,                                             #  number of layers of latent variables.
                       layer_latent_shape=100,                                   #  shape of the latent features of each layer.
                       data_layer_type="conv2d",                                 #  type of cells for the data distribution.
                       posterior_layer_type="conv2d",                            #  type of cells for the posterior distribution.
                       prior_layer_type="conv2d",                                #  type of cells for the prior distribution.
                       residual_var_layer=True,                                  #  flag indicating a residual variational layer.
                       preproc_encoder_type='resnet',                            #  type of cells for the first encoder.
                       encoder_type='resnet',                                    #  type of cells for the  encoder.
                       postproc_decoder_type='resnet',                           #  type of cells for the post-processing cell in encoder.
                       decoder_type='resnet',                                    #  type of cells for the decoder.
                       merge_encoder_type='conv2d_sum',                          #  type of concatenation cell in the encoder.
                       merge_decoder_type='concat_conv2d',                       #  type of concatenation cell in the decoder. 
                       num_proc_blocks=1,                                        #  number of preprocessing and postprocessing blocks.                  
                   )
        
    @staticmethod
    def get_default_network_hparams(h):
        
        ## hyperparameters for the networks responsible for generating the distributional parameters
        
        prior_net_hparams={"conv2d":"kernel_size=1,num_times=1,activation=elu"}
        
        posterior_net_hparams={"conv2d":"kernel_size=3,num_times=1,activation="}
        
        init_posterior_net_hparams={"conv2d":"kernel_size=3,num_times=2,activation=elu"}
        
        data_net_hparams={"conv2d": "kernel_size=3,num_times=1,activation=elu"}
        
        
        ## network hyperparaneters for the encoder
        
        preproc_encoder_hparams={"resnet": "use_stem=True,activation=swish,use_batch_norm=True"}
        
        encoder_hparams={"resnet": "use_stem=False,activation=swish,use_batch_norm=True"}
        
        
        ## network hyperparaneters for the decoder
        
                
        postproc_decoder_hparams={"resnet": "use_stem=False,activation=swish,use_batch_norm=True"}
        
        decoder_hparams={"resnet": "use_stem=False,activation=swish,use_batch_norm=True"}
        
        return HParams(## hyperparameters for the networks responsible for generating the distributional parameters
                       prior= LAYER[h.prior_layer_type].get_default_hparams().update_config(prior_net_hparams[h.prior_layer_type]),    
                       posterior= LAYER[h.posterior_layer_type].get_default_hparams().update_config(posterior_net_hparams[h.posterior_layer_type]),
                       init_posterior= LAYER[h.posterior_layer_type].get_default_hparams().update_config(init_posterior_net_hparams[h.posterior_layer_type]),
                       data= LAYER[h.data_layer_type].get_default_hparams().update_config(data_net_hparams[h.data_layer_type]),
                       ## hyperparaneters for the encoder
                       preproc_encoder= LAYER[h.preproc_encoder_type].get_default_hparams().update_config(preproc_encoder_hparams[h.preproc_encoder_type]),
                       encoder= LAYER[h.encoder_type].get_default_hparams().update_config(encoder_hparams[h.encoder_type]),
                       ## hyperparaneters for the decoder
                       postproc_decoder= LAYER[h.postproc_decoder_type].get_default_hparams().update_config(postproc_decoder_hparams[h.postproc_decoder_type]),
                       decoder = LAYER[h.decoder_type].get_default_hparams().update_config(decoder_hparams[h.decoder_type]),
        )







