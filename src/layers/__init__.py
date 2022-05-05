# ============================================================================
#  Copyright 2021. 
#
#
#  Author:  Ifigeneia Apostolopoulou 
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


"""""
    ***************************************************
    
    ****** Deep Attentive Variational Inference. ******
    
    ***************************************************
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from layers.resnet import Conv2D
from layers.resnet import ConcatConv2D
from layers.resnet import Conv2DSum
from layers.resnet import ResNet
from layers.resnet import FactorizedReduce
from layers.resnet import SqueezeAndExciteBlock
from layers.resnet import LAYER
from layers.attention import DepthWiseAttention
from layers.attention import NonlocalResNetBlock
from layers.attention import ALIGNMENT_FUNCTION
from layers.util import ACTIVATION
from layers.util import REGULARIZER
from layers.weight_norm import  WeightNorm
from layers.resnet import weightnorm_layer

__all__ = [
           'ACTIVATION',
           'LAYER',
           'Conv2D',
           'ConcatConv2D',
           'Conv2DSum',
           'ResNet',
           'FactorizedReduce',
           'SqueezeAndExciteBlock',
           'DepthWiseAttention',
           'NonlocalResNetBlock',
           'ALIGNMENT_FUNCTION',
           'WeightNorm',
           'weightnorm_layer',
           ]



