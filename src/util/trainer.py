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


#

"""
    ***************************************************
    
    ****** Deep Attentive Variational Inference. ******
    
    ***************************************************
"""

import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger,LambdaCallback

import os
import sys
import csv

from scipy.io import loadmat
from util.learning_rate_schedule import LearningRateScheduler,LEARNING_RATE_SCHEDULER
from util.data import DatasetSequence
from util.eval import create_eval_command
from util.hparams import HParams


__all__ = [
           'keras_train',
           ]


def keras_train(model,
                dataset_transform,
                hparams,
                checkpoint_period,
                train_log_dir,
                train_log_subdir=None,
                epoch=0,
                mirrored_strategy=None,
                dataset_builder=None,
                dataset_path='',
                image_info=None,
                ):
    
    """
        Utility function that trains a deep variational autoencoder.
        
        Args:
            model:                  A VAE model to be trained.
            dataset_builder:        A tfds.builder (refer to tensorflow_datasets for the detailed documentation) object.
            dataset_path:           Path of user's dataset if  tfds.builder=='None'.
            dataset_transform:      The transformations to be applied on the dataset.
            hparams:                The training parameters.
            checkpoint_period:      Save period of the model during training.
            train_log_dir:          Directory for the training logs.
            train_log_subdir:       Sub-directory name in train_log_dir for the training logs.
            epoch:                  Epoch from which the training is resumed (if a model_filepath is provided).
            mirrored_strategy:      A tf.distribute.MirroredStrategy in case of a multi-gpu model.
            image_info:             Dictionary with info about the image to be generated.
                                        shape:Rank-3 shape of the image data to be generated.
                                        low: float number indicating the smallest possible value of the data.
                                        high: float number indicating the largest possible value of the data.
                                        pad_size:: integer number indicating the number of pixels to be croped from the generated image, otherwise non-positive.
        Output:
            final_model.h5:         Saved model in .5 format. in the specified sub-directory.
            keras_trainer_log.csv:  The learning curves of the deep model that include: [total loss (including regularization), negative log-likelihood, kl divergence, nelbo]
            eval_command.sh:        Command for loading and evaluation the model.
            train_command.sh:       Comand for training a new model from scratch.
        
    """
    
    
    # load dataset
    
    if dataset_builder is not None:
        dataset_builder.download_and_prepare()
        dataset = dataset_builder.as_dataset()    
        train_dataset, test_dataset = dataset["train"], dataset["test"]
    
        assert isinstance(train_dataset, tf.data.Dataset)
    
        
        # extract training samples
        train_dataset = train_dataset.batch(dataset_builder.info.splits['train'].num_examples)
        train_features = tf.compat.v1.data.make_one_shot_iterator(train_dataset).get_next()
        train_images = train_features['image']    

    
    
        # extract test samples
        test_dataset = test_dataset.batch(dataset_builder.info.splits['test'].num_examples)
        test_features = tf.compat.v1.data.make_one_shot_iterator(test_dataset).get_next()
        
        test_images = test_features['image']
        test_images=tf.dtypes.cast(test_images, tf.float32)
        
    else:
        images = loadmat(dataset_path)
        image_shape=train_images=image_info['shape']
        train_images=tf.transpose(images['data'])
        train_images=tf.reshape(train_images,[-1, image_shape[0], image_shape[1], image_shape[2]])
               
        test_images=tf.transpose(images['testdata'])
        test_images=tf.dtypes.cast(test_images, tf.float32)
        test_images=tf.reshape(test_images,[-1, image_shape[0], image_shape[1], image_shape[2]])
        

    # create training directory
    train_log_dir=create_train_log_dir(train_log_dir, train_log_subdir=model.name if not train_log_subdir else  train_log_subdir)


    ## set-up training checkpoint    
    checkpoint=ModelCheckpoint(save_freq=checkpoint_period,save_dir=train_log_dir)


    ## set-up learning rate scheduler, if any
    
    hparams['lr_steps_per_epoch']=int(train_images.shape[0]/(hparams.num_gpus*hparams.batch_size))
    hparams['lr_decay_epochs']=hparams.epochs-hparams.lr_warmup_epochs
    
    lr_callback=[LearningRateScheduler(LEARNING_RATE_SCHEDULER[hparams.learning_rate_schedule](initial_learning_rate=hparams.learning_rate,**{key:getattr(hparams, 'lr_'+key) for key in LEARNING_RATE_SCHEDULER[hparams.learning_rate_schedule].__init__.__code__.co_varnames[2:]}))] if hparams.learning_rate_schedule!='constant' else []
    
    ## set-up kl- annealing scheduler, if any
    kl_scheduler=KLRegularizerSchedule(hparams.kl_scheduler)
    

    ##create loss function
    loss=[lambda y1, y2: y2]*(3)
        
    loss_weight=[ 1.0]+[kl_scheduler.beta]+[0.0]

        
    # compile model

    kwargs = {key:getattr(hparams, 'keras_lr_'+key) for key in OPTIMIZER[hparams.optimizer].__init__.__code__.co_varnames[2:-1]}
    optimizer=OPTIMIZER[hparams.optimizer](learning_rate=hparams.learning_rate,**kwargs)


    with mirrored_strategy.scope():
        model.compile(optimizer=optimizer,
                            loss=loss,
                            loss_weights=loss_weight,
                            )


    # set-up image datasets
    y=[tf.zeros(train_images.shape[0])]*(3)
    y_test=[tf.zeros(test_images.shape[0])]*(3)
        
        
    test_images=dataset_transform.apply(test_images,training=False)
        
    data=DatasetSequence(x_set=train_images, y_set=y, batch_size=hparams.batch_size*hparams.num_gpus, shape=model.image_shape,transform=dataset_transform)


    cp_callback = tf.keras.callbacks.ModelCheckpoint(train_log_dir+"cp.ckpt",
                                                         save_weights_only=True,
                                                         verbose=1)
    
    # start training
    model.fit(x=data,
                  validation_data=(test_images,y_test),
                  epochs=epoch+hparams.epochs,
                  initial_epoch=epoch,
                  callbacks=[LambdaCallback(on_epoch_begin=lambda epoch, logs: data.on_epoch_begin())]+[CSVLogger(train_log_dir+'keras_trainer_log.csv',append=True)]+[checkpoint,lr_callback,kl_scheduler],
                  )


    #save model weights
    model.save_weights(train_log_dir+"final_model.h5")





OPTIMIZER={"adam":tf.keras.optimizers.Adam,
           "adamax":tf.keras.optimizers.Adamax,
           "rmsprop":tf.keras.optimizers.RMSprop,
}

############  utilitity functions   ############
################################################


def get_default_train_hparams():
    return HParams(
                   num_gpus=1,                                                # number of available gpus
                   batch_size=256,                                            # batch_size per gpu
                   epochs=400,                                                # number of training epochs
                   kl_scheduler= KLRegularizerSchedule.get_default_hparams(), # parameters for the kl-annealing
                   ## parameters for the learning rate scheduler
                   learning_rate=1e-2,                                        # initial learning rate 
                   learning_rate_schedule='cosine_warmup',                    # learning rate scheduler
                   lr_name='learning_rate_scheduler',                         # name of the scheduler.
                   lr_decay_epochs=1000,                                      # number of epochs, after the warmup, during which the learning rate decreases from learning_rate to lr_min_learning_rate at the end of each epoch
                   lr_steps_per_epoch=1563,                                   # the learning steps per gpu, applied during a epoch.
                   lr_verbose=False,                                          # flag indicating whether the current learning rate will be printed
                   lr_warmup_epochs=5,                                        # number of epochs, during which the learning rate increases from lr_min_learning_rate to learning_rate per gradient step
                   lr_min_learning_rate=0.0001,                               # minimum learning rate.
                   lr_first_decay_epochs=100,                                 # number of epochs, after the warmup, and for the first period of the restarts, during which the learning rate decreases from learning_rate to lr_min_learning_rate at the end of each epoch
                   lr_t_mul=2.0,                                              # how many times the decay will last in the current restart
                   lr_m_mul=1.0,                                              # how much smaller (how many times smaller), compared to initial_learning_rate, the learning rate will be at the beginning of the restart.
                   ## parameters for the optimizer.
                   ## refer to the documentation of tf.keras.optimizers.Adam for a detailed description.
                   optimizer="adamax",
                   keras_lr_momentum=0.0001,
                   keras_lr_beta_1=0.9,                                        # the exponential decay rate for the 1st moment estimates.
                   keras_lr_beta_2=0.999,                                      # the exponential decay rate for the 2nd moment estimates. 
                   keras_lr_epsilon=1e-03,                                     # small constant for numerical stability.
                   keras_lr_kwargs='',
                   keras_lr_name='',
                   )


def create_train_log_dir(train_log_dir, train_log_subdir=None):
    """ 
        Utility function that creates a unique subdirectory in train_log_dir with postfix name train_log_subdir.
    """
    
    if train_log_subdir:
        i = 0
        while os.path.exists(train_log_dir+str(i)+"_"+train_log_subdir+"/"):
            i += 1

        train_log_dir=train_log_dir+str(i)+"_"+train_log_subdir+"/"
        os.makedirs(train_log_dir, exist_ok=True)

    # save the command line arguments
    f= open(train_log_dir+"train.sh","w")
    f.write('python3 '+' '.join(sys.argv[0:]))
    f.close()

    # create the command for the model evaluation
    create_eval_command(sys.argv[0:],train_log_dir)
    

    return train_log_dir



class ModelCheckpoint(tf.keras.callbacks.Callback):
    """
        Utility class for periodically saving the model.
    """
    
    def __init__(self, save_freq, save_dir):        
        self.save_freq = save_freq
        
        self.save_dir=save_dir
    
    def on_epoch_end(self, epoch, logs=None):
        i=1
#         if (epoch+1) % self.save_freq == 0: 
#             
#             self.model.save_weights("./"+self.save_dir+"checkpoint_model_"+str(epoch)+".h5")



class KLRegularizerSchedule(tf.keras.callbacks.Callback):
    
    """
        Utility function for kl-annealing.
        
        Returns the kl-regularization term  for the current epoch needed for the deterministic warm-up described in the papers:
        
        References:
        
            [1]. Sønderby, C.K., Raiko, T., Maaløe, L., Sønderby, S.K. and Winther, O., 2016. Ladder variational autoencoders.
                 In Advances in neural information processing systems (pp. 3738-3746).
            [2]. Sønderby, C.K., Raiko, T., Maaløe, L., Sønderby, S.K. and Winther, O., 2016.
                 How to train deep variational autoencoders and probabilistic ladder networks.
                In 33rd International Conference on Machine Learning (ICML 2016),
            [3]. Bowman SR, Vilnis L, Vinyals O, Dai AM, Jozefowicz R, Bengio S.
                 Generating sentences from a continuous space. arXiv preprint arXiv:1511.06349. 2015 Nov 19
        
    """
    def __init__(self, hparams):
        self.beta = tf.Variable(0.0)

        self.hparams = hparams
     
    def on_epoch_begin(self, epoch, logs=None):
        if epoch<self.hparams.warmup_epochs:
            beta= epoch*((1.0-self.hparams.warmup_smooth_start)/self.hparams.warmup_epochs)+self.hparams.warmup_smooth_start
        else:
            beta= 1.0
            
        tf.keras.backend.update(self.beta,beta)

    @staticmethod
    def get_default_hparams():
        return HParams(
                       warmup_epochs=16,         # number of  KL warm-up epochs (KL regularization term <1), see [2].
                       warmup_smooth_start=0.1,  # offset value for linear warm-up (start value of the kl-coefficient for epoch=0).
                       )


