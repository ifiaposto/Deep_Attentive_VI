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

from util.hparams import HParams

__all__ = [
           'DatasetSequence',
           'DatasetTransform',
           ]

class DatasetTransform():
    
    def __init__(self,hparams,info):
        
        self._hparams=hparams
        self._info=info
        
    def apply(self,x,training=True):
        
        
        if training and self._hparams.flip_horizontally:
            x=tf.image.random_flip_left_right(x)
             
        if training and self._hparams.flip_vertically:
            x=tf.image.random_flip_up_down(x)
            
        if self._hparams.pad_size>0:
            paddings=tf.constant([[0,0],[self._hparams.pad_size, self._hparams.pad_size,], [self._hparams.pad_size, self._hparams.pad_size],[0,0]])
                        
            x=tf.pad(x, paddings)
            


        if self._hparams.dequantize:
            x = ((x- self._info['low']) / (self._info['high'] - self._info['low']))
       

        
        if self._hparams.sample_binary:
            p = tfp.distributions.Bernoulli(probs=x)
            
            x=p.sample()
            
            
        x=tf.cast(x,dtype=tf.float32)
            
            
        
        return x
    
    @staticmethod
    def get_default_hparams(dataset_name):
        
        if dataset_name=='cifar10':
            return HParams(flip_horizontally=True,
                           flip_vertically=True,
                           sample_binary=False,
                           dequantize=False,
                           pad_size=0,
                       )
            
        if dataset_name=='omniglot':
            return HParams(flip_horizontally=False,
                           flip_vertically=False,
                           sample_binary=True,
                           dequantize=False,
                           pad_size=2,
                       )


        if dataset_name=='mnist':
            return HParams(flip_horizontally=False,
                           flip_vertically=False,
                           sample_binary=True,
                           dequantize=True,
                           pad_size=2,
                       )
            

    
class DatasetSequence(tfk.utils.Sequence):
    def __init__(self, x_set, y_set, shape, batch_size, transform,training=True,shuffle=True):
        
        self.original_x, self.y = x_set, y_set      
        self.x=tf.convert_to_tensor(self.original_x)
                
                
        self.batch_size = batch_size
        
    
        self.size=x_set.shape[0]

        self.shape=shape
        
        self.transform=transform
        
        self._training=training
        
        self.shuffle=shuffle
        
        self.indices = np.arange(self.size)
        

    

    def __len__(self):
        return int(np.ceil(self.size / float(self.batch_size)))
    
    def on_epoch_begin(self):
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        

        
        inds = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x =tf.convert_to_tensor(tf.gather(self.x,inds,axis=0))
    
        batch_x=self.transform.apply(batch_x,self._training)
        
        if self.y is not None:

            batch_y = [tf.convert_to_tensor(y_i[idx * self.batch_size:(idx + 1) * self.batch_size]) for y_i in self.y]
        
            return batch_x, batch_y
        return batch_x




