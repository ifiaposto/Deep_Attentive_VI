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



from model.vae  import DeepVAE
from model.variational_layer import VariationalLayer
from model.data_layer import DataLayer




__all__ = [
    'DeepVAE',
    'VariationalLayer',
    'DataLayer',
]


