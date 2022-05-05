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

import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K
import tensorflow as tf
import numpy as np


__all__ = [
           'CosineWarmup',
           'LEARNING_RATE_SCHEDULER',
           'LearningRateScheduler',
           ]

class LearningRateScheduler(tf.keras.callbacks.Callback):
    def __init__(self, schedule, verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.global_step=0
    def on_train_batch_begin(self, batch, logs=None):
        
        self.global_step=self.global_step+1
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        try:  # new API
            lr = float(K.get_value(self.model.optimizer.lr))
            lr = self.schedule(self.global_step, lr)
        except TypeError:  # Support for old API for backward compatibility
            lr = self.schedule(self.global_step)
        if not isinstance(lr, (ops.Tensor, float, np.float32, np.float64)):
            raise ValueError('The output of the "schedule" function '
                       'should be float.')
        if isinstance(lr, ops.Tensor) and not lr.dtype.is_floating:
            raise ValueError('The dtype of Tensor should be float')
        K.set_value(self.model.optimizer.lr, K.get_value(lr))
        if self.verbose > 0:
            print('\nGlobal step %05d: LearningRateScheduler reducing learning '
            'rate to %s.' % (self.global_step + 1, lr))

@keras_export("keras.optimizers.schedules.CosineRestartWarmup")
class CosineRestartWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
      See [Loshchilov & Hutter, ICLR2016](https://arxiv.org/abs/1608.03983),
    """
    
    def __init__(
        self,
        initial_learning_rate,
        min_learning_rate,
        warmup_epochs,
        first_decay_epochs,
        steps_per_epoch,
        t_mul=2.0,
        m_mul=1.0,
        verbose=True,
        name=None
        
        ):
        
        self._min_learning_rate=min_learning_rate
        self._initial_learning_rate = initial_learning_rate
        self.first_decay_epochs = first_decay_epochs
        self.curr_learning_rate=self._min_learning_rate
        self._delta=self._initial_learning_rate-self._min_learning_rate
        
        self._warmup_epochs=warmup_epochs
        self._warmup_steps=warmup_epochs*steps_per_epoch
        self.steps_per_epoch=steps_per_epoch
            
        
        
        self._t_mul = t_mul
        self._m_mul = m_mul
        
        self.alpha= self._min_learning_rate/self._initial_learning_rate
        
        
        self.name=name
        
        self.verbose=verbose
        
    def __call__(self, step):
        with ops.name_scope_v2(self.name or "CosineRestartWarmup"):
        
            self.curr_learning_rate = ops.convert_to_tensor_v2(self.curr_learning_rate, name="current_learning_rate")
                
            dtype =  self.curr_learning_rate.dtype
                
            epoch=step//self.steps_per_epoch
                
            if step<=self._warmup_steps:
                self.curr_learning_rate=self._min_learning_rate+float(step/self._warmup_steps)*self._delta
                if self.verbose:
                    print('epoch: %d, warming learning rate: %s'%(epoch,tf.keras.backend.eval(self.curr_learning_rate)))
                    
                return self.curr_learning_rate
                
            local_epoch=epoch-self._warmup_epochs
            
            
            
            def compute_step(completed_fraction, geometric=False):
                """Helper for `cond` operation."""
                if geometric:
                    i_restart = tf.floor(tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /tf.math.log(t_mul))

                    sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                    completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

                else:
                    i_restart = tf.floor(completed_fraction)
                    completed_fraction -= i_restart

                return i_restart, completed_fraction
            
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)
            first_decay_epochs = tf.cast(self.first_decay_epochs, dtype)
            
            completed_fraction = local_epoch / first_decay_epochs
            i_restart, completed_fraction = tf.cond(tf.equal(t_mul, 1.0),
                                                    lambda: compute_step(completed_fraction, geometric=False),
                                                    lambda: compute_step(completed_fraction, geometric=True))

            m_fac = m_mul**i_restart
            cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(tf.constant(math.pi) * completed_fraction))
            decayed = (1 - alpha) * cosine_decayed + alpha

            self.curr_learning_rate=tf.multiply(self._initial_learning_rate, decayed, name=self.name)
            
            
            if self.verbose:
                print('step: %d, epoch: %d, cosine decaying learning rate: %s'%(step,epoch,tf.keras.backend.eval(self.curr_learning_rate)))

            return self.curr_learning_rate
    
    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_epochs": self.first_decay_epochs,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "name": self.name
        }

@keras_export("keras.optimizers.schedules.CosineWarmup")
class CosineWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(
                 self,
                 initial_learning_rate,
                 min_learning_rate,
                 warmup_epochs,
                 decay_epochs,
                 steps_per_epoch,
                 verbose=True,
                 name=None):
            
            super(CosineWarmup, self).__init__()
            
            
            self._min_learning_rate=min_learning_rate
            
            self._initial_learning_rate=initial_learning_rate
            
            self.curr_learning_rate=self._min_learning_rate
            
            self.alpha= self._min_learning_rate/self._initial_learning_rate
            
            self._delta=self._initial_learning_rate-self._min_learning_rate
            
            self._warmup_epochs=warmup_epochs
            
            self._warmup_steps=warmup_epochs*steps_per_epoch
            
            self.name=name
            
            self._decay_epochs=decay_epochs
            
            self.steps_per_epoch=steps_per_epoch
            
            self.verbose=verbose
            

                

            
        def __call__(self, step):
            with ops.name_scope_v2(self.name or "CosineWarmup"):
                
                self.curr_learning_rate = ops.convert_to_tensor_v2(self.curr_learning_rate, name="current_learning_rate")
                
                dtype =  self.curr_learning_rate.dtype
                
                decay_epochs = math_ops.cast(self._decay_epochs, dtype)
                
                epoch=step//self.steps_per_epoch
                
                if step<=self._warmup_steps:
                    self.curr_learning_rate=self._min_learning_rate+float(step/self._warmup_steps)*self._delta
                    if self.verbose:
                        print('epoch: %d, warming learning rate: %s'%(epoch,tf.keras.backend.eval(self.curr_learning_rate)))
                    
                    return self.curr_learning_rate
                
                local_epoch=epoch-self._warmup_epochs
                
                
                
                if local_epoch>decay_epochs:
                    if self.verbose:
                        print('step: %d, epoch: %d, cosine decaying learning rate: %s'%(step,epoch,tf.keras.backend.eval(self.curr_learning_rate)))

                    return self.curr_learning_rate
                
                completed_fraction = local_epoch / decay_epochs
                
                cosine_decayed = 0.5 * (1.0 + tf.cos(tf.constant(math.pi) * completed_fraction))

                decayed = (1 - self.alpha) * cosine_decayed + self.alpha
                
                self.curr_learning_rate=tf.multiply(self._initial_learning_rate, decayed)
                
               
                if self.verbose:
                    print('step: %d, epoch: %d, cosine decaying learning rate: %s'%(step,epoch,tf.keras.backend.eval(self.curr_learning_rate)))

                return self.curr_learning_rate
            
        def get_config(self):
            return {"initial_learning_rate": self.initial_learning_rate,
                    "min_learning_rate": self.min_learning_rate,
                    "warmup_steps": self.warmup_steps,
                    "steps_per_epoch":self.steps_per_epoch,
                    "name": self.name,
                    }   




LEARNING_RATE_SCHEDULER = {"cosine_warmup":CosineWarmup, 
                           "cosine_restart_warmup":CosineRestartWarmup 
                           }

