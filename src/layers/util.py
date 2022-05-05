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

import tensorflow.keras as tfk
from tensorflow.python.ops import nn
from tensorflow.keras.initializers import  VarianceScaling
from tensorflow.python.util.tf_export import keras_export

__all__ = [
           'ACTIVATION',
            'REGULARIZER'
            'DeepVaeInit'
           ]

@keras_export('keras.initializers.deep_vae',
              'keras.initializers.deepVae',
              v1=[])

## weight initializer

class DeepVaeInit(VarianceScaling):
    """ Weight Initializer for deep VAE. """
    def __init__(self, seed=None):
        super(DeepVaeInit, self).__init__(scale=0.33, mode='fan_in', distribution='uniform', seed=seed)
    def get_config(self):
        return {'seed': self.seed}    
    
## activation function 

def swish(x):
    """Swish activation function.
        Arguments:
        x: Input tensor.
        Returns:
        The swish activation applied to `x`.
        """
    return nn.swish(x)
    

ACTIVATION={"tanh":tfk.activations.tanh,
            "relu":tfk.activations.relu,
            "softmax":tfk.activations.softmax,
            "sigmoid":tfk.activations.sigmoid,
            "softplus":tfk.activations.softplus,
            "leaky_relu":tfk.layers.LeakyReLU,
            "exp":tfk.activations.exponential,
            "elu":tfk.activations.elu,
            "swish":swish,
}
## weight regularizers

REGULARIZER={"l1":tfk.regularizers.l1,"l2":tfk.regularizers.l2}


