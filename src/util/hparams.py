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

__all__ = [
           'HParams',
           ]

import re

class HParams(object):
    """Dictionary of hyperparameters."""
    def __init__(self, **kwargs):
        self.dict_ = kwargs
        self.__dict__.update(self.dict_)
    
    def update_config(self, in_string):
        
        """Update the dictionary with a comma separated string."""

        if not in_string: return self        
        pairs=re.split(r',\s*(?![^[]*\])', in_string)
   


        pairs = [re.split(r'=\s*(?![^[]*\])', pair) for pair in pairs]

        for key, val in pairs:
            
            if isinstance(self.dict_[key], bool):

                self.dict_[key] = val=='True'
            elif isinstance(self.dict_[key], HParams):

                self.dict_[key].update_config(val.strip('[]'))
            else:
                val = val.strip('][').split(',')

                if len(val)>1:
                    val= [float(v) for v in val]
                    if  val[0].is_integer:
                        self.dict_[key]=[int(v) for v in val]
                    else:
                        self.dict_[key]=val
                else:
                    self.dict_[key]=type(self.dict_[key])(val[0])
        self.__dict__.update(self.dict_)
        
        return self

    def save(self,w):
        for key, val in self.dict_.items():
            if isinstance(self.dict_[key], HParams):
                w.writerow([key])
                val.save(w)
            else:
                w.writerow([key, val])

    def extract(self,keys):
        """ returns a new dictionary that contrains only the pairs with key contained in keys.
            It ignores keys that do not exist in the dictionary.
            """
        return { x: self[x] for x in keys if self.exists(x)}
    
    def keys(self):
        """ returns the names of the hyperparameters"""
    
        return list(self.__dict__)

    def exists(self,key):
        """ checks whether key exists in the keys of the dictionary"""
        return key in list(self.__dict__)


    def __getitem__(self, key):
        return self.dict_[key]
    
    def __setitem__(self, key, val):
        self.dict_[key] = val
        self.__dict__.update(self.dict_)




