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
from stat_tools import DISTRIBUTION
from layers import LAYER
import copy
import numpy as np

__all__ = ['DataLayer',
           ]



class DataLayer(tf.keras.layers.Layer):
    """
        *** Data Distribution ****
        
        Args:
          
            preproc_hparams:    Hyper-parameters for the residual cell that pre-processes the condition of the data distribution.
            data_dist_hparams:  Parameters for the data distribution.
            network_hparams:    Network hyper-parameters of the cell generating parameters of the datadistribution.
            class_dist:         Class of data distribution.
            num_proc_blocks:    Number of pre-processing blocks.
            image_info:         Dicitionary with info about the image to be generated.
                                        shape:Rank-3 shape of the image data to be generated.
                                        low: Float number indicating the smallest possible value of the data.
                                        high: Float number indicating the largest possible value of the data.
                                        crop_output: Integer number indicating the number of pixels to be croped from the generated image, otherwise non-positive.
    
    """
    def __init__(self,
                 preproc_hparams=None,
                 data_dist_hparams=None,
                 network_hparams=None,
                 class_dist=None,
                 image_info=None,
                 num_proc_blocks=1,
                 **kwargs):
        
        
        super(DataLayer, self).__init__(dtype=tf.float32,**kwargs)
        
        # set hyper-parameters
        
        self._image_info=image_info
        
        self._preproc_hparams=preproc_hparams
        
        
        self._data_dist_hparams=data_dist_hparams
        
        self._network_hparams=network_hparams
        
        self._class_dist=class_dist
        
        self._num_proc_blocks=num_proc_blocks
        
        # build preprocessing layer
        self._preproc_layer=[]
        preproc_hparams=copy.deepcopy(preproc_hparams)
        preproc_hparams['num_filters']=self._preproc_hparams['num_filters']*self._num_proc_blocks
        for s in range(self._num_proc_blocks):
                        
            if self._preproc_hparams is not None:
                self._preproc_layer.append(LAYER[self._preproc_hparams.layer_type](hparams=preproc_hparams))
            else:
                self._preproc_layer.append(tf.keras.layers.Lambda(lambda x:x))
                
            if s+1<self._num_proc_blocks:
                preproc_hparams=copy.deepcopy(preproc_hparams)
                preproc_hparams['num_filters']=preproc_hparams['num_filters']/2
                           
        #build data distribution        

        image_width=self._image_info['shape'][0]+self._image_info['pad_size']*2
        image_height=self._image_info['shape'][1]+self._image_info['pad_size']*2
        image_shape=[image_width, image_height,self._image_info['shape'][-1]]

        self._data_dist=DISTRIBUTION[self._class_dist](shape=image_shape,
                                                                      hparams= self._data_dist_hparams,
                                                                      network_hparams=self._network_hparams,
                                                                      )


    def generate(self,latent_codes):
        """
            It generates image data given the latent factors.
            
            Inputs:
                latent_codes:       4D tensor of stochastic context.     
            Returns:
                y:                  4D tensor of the generated images.
            
        """
        with tf.name_scope(self.name or 'DataLayer_generate'):
            
            #preprocess latent codes
            for s in range(self._num_proc_blocks):
                
                latent_codes=self._preproc_layer[s](latent_codes,training=False)
                

            ## generate image from latent codes, 1 sample per datapoint in the batch
            
            y =self._data_dist.sample(condition=latent_codes,training=False)
            
            return y
        
    def count_params(self):
        """ 
            Utility function that counts the number of the trainable parameters for the module.
        """
        postproc_params=0
        
        for s in range(self._num_proc_blocks):
         
            postproc_params_s=int(np.sum([tfk.backend.count_params(p) for p in self._preproc_layer[s].trainable_weights]))
            postproc_params=postproc_params+postproc_params_s

        data_dist_params=int(np.sum([tfk.backend.count_params(p) for p in self._data_dist.trainable_weights]))
        
        return data_dist_params+postproc_params
        
        
    def decode(self,latent_codes, x=None,training=True):
        """
            It computes the negative, data-loglikelihood.
            
            Inputs:
                latent_codes: 4D tensor of stochastic context.    
                x:            4D tensor of images.
                training:     Boolean or boolean scalar tensor, indicating whether to run the Network in training mode or inference mode. 
               
                
            Returns:
                the conditional log-likelihood log p(x|latent_codes).
                
        """
        with tf.name_scope(self.name or 'DataLayer_decode'):
            
            for s in range(self._num_proc_blocks):
                
                latent_codes=self._preproc_layer[s](latent_codes,training)
            
            
            # call the image-conditional to get the conditional data likelihood
            cond_ll=self._data_dist.log_prob(x=x,condition=latent_codes,training=training,crop_size=self._image_info['pad_size'])
            
            return tf.math.negative(cond_ll)
            
           







