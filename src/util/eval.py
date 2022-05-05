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


import matplotlib.pyplot as plt
import os
import csv
import numpy as np

from scipy.io import loadmat
from util.data import *
from util.hparams import *
import tensorflow as tf



__all__ = [
            'keras_eval'
           ]

def plot_and_save(image,image_shape,f_name):
    
        """ Utility function for ploting images """
        
        image=np.reshape(image,image_shape)
        plt.figure(figsize=(10, 10))
        plt.imshow(image[:,:,0] if image_shape[-1]==1 else image,cmap='Greys_r' if  len(image_shape)==2 or image_shape[-1]==1 else 'viridis')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.savefig(f_name, bbox_inches = 'tight',pad_inches = 0)


def keras_eval (model,
                model_filepath,
                hparams,
                eval_log_dir,
                dataset_transform,
                dataset_builder=None,
                dataset_path='',
                image_info=None,
                mirrored_strategy= None,
                ):
    """
         Utility function that evaluates a deep variational autoencoder.
         
         Args:
            model:                  A VAE model to be evaluated.
            model_filepath:         Filename with the model weights.
            dataset_builder:        A tfds.builder (refer to tensorflow_datasets for the detailed documentation) object.
            dataset_path:           Path of user's dataset if  tfds.builder==None.
            hparams:                Parameters for the model evaluation.
            eval_log_dir:           Directory for the evaluation logs.
            dataset_transform:      The transformations to be applied on the dataset.
            mirrored_strategy:      A tf.distribute.MirroredStrategy in case of a multi-gpu model.
            image_info:             Dictionary with info about the image to be generated.
                                        shape:Rank-3 shape of the image data to be generated.
                                        low: float number indicating the smallest possible value of the data.
                                        high: float number indicating the largest possible value of the data.
                                        pad_size:: integer number indicating the number of pixels to be croped from the generated image, otherwise non-positive.
        Returns:
            In the eval_log_dir:
                eval_test_log.csv: 
                    --- if hparams.compute_logp=True:  [total_loss (including regularization penalty), nll (the conditional negative log-likelihood), kl (kl-loss across all layers), neblbo, marg logp (marginal loglikelihood computed by importance sampling)] 
                    --- if hparams.compute_logp=False: [total_loss, nll, kl, nelbo,0.0]
                sample_image_x.png: samples images drawn from the model if hparams.generate=True.
                [original_test_image_x.png,reconstructed_test_image_x.png]: original vs reconstructed image (from the latent codes) if hparams.reconstruct=True.
    """


    
    # load dataset
        
    if dataset_builder is not None:
        dataset_builder.download_and_prepare()
        dataset = dataset_builder.as_dataset()    
        test_dataset = dataset["test"]
    
        assert isinstance(test_dataset, tf.data.Dataset)
    
        # extract test samples
        test_dataset = test_dataset.batch(dataset_builder.info.splits['test'].num_examples)
        test_features = tf.compat.v1.data.make_one_shot_iterator(test_dataset).get_next()
        
        test_images = test_features['image']
        test_images=tf.dtypes.cast(test_images, tf.float32)
            
    else:
        images = loadmat(dataset_path)
        image_shape=image_info['shape']


        test_images=tf.transpose(images['testdata'])
        test_images=tf.dtypes.cast(test_images, tf.float32)
        test_images=tf.reshape(test_images,[-1, image_shape[0], image_shape[1], image_shape[2]])



   
    small_batch=dataset_transform.apply(test_images[0:2,:,:,:],training=False)
    with mirrored_strategy.scope():
            model.call(inputs=tf.dtypes.cast(small_batch, tf.float32),training=False)
            model.built = True
    model.load_weights(model_filepath)
        

 
    if hparams.compute_logp:
                test_loss=model.predict(DatasetSequence(x_set=test_images, y_set=None,transform=dataset_transform, batch_size=hparams.batch_size*hparams.num_gpus, shape=model.image_shape,training=False),
                                        verbose=hparams.verbose,
                                        num_samples=hparams.num_importance_samples,
                                        workers=hparams.workers
                                        )
                
                test_loss=tf.reduce_mean(test_loss,0)
       
                with open(eval_log_dir+"eval_test_log.csv", 'a', newline='') as csv_file:
                    csv_eval_test_logger = csv.writer(csv_file, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    csv_eval_test_logger.writerow(['marg logp'])
                    csv_eval_test_logger.writerow([test_loss])

              
    # generate new sample images
    if hparams.generate:
            
        _,y=model.generate(num_samples=hparams.num_sample_images)
             
        for k in range(hparams.num_sample_images):
            plot_and_save(y[k],model.image_shape,os.path.join(eval_log_dir+'sample_image_%d.png' % (k)))
               
               
      
    # reconstruct test images from the latent codes drawn from the variational model.
    if hparams.reconstruct:
        test_images=dataset_transform.apply(test_images,training=False)
        if hparams.shuffle:
            shuffled_indices = tf.random.shuffle(tf.range(start=0, limit=tf.shape(test_images)[0], dtype=tf.int32))
            shuffled_indices=shuffled_indices[:hparams.num_test_images,...]
            test_images=tf.gather(test_images, shuffled_indices,axis=0)    
            x_reconstr=model.reconstruct(test_images)
               
        for k in range(hparams.num_test_images):
               
            image=np.reshape(test_images[k],model.image_shape)
            # in [0,255]
            image=tf.cast(image,dtype=tf.int32)
            plot_and_save(image,model.image_shape,os.path.join(eval_log_dir+'original_test_image_%d.png' % (k)))
            
            
            image=np.reshape(x_reconstr[k],model.image_shape)
            plot_and_save(image,model.image_shape,os.path.join(eval_log_dir+'reconstructed_test_image_%d.png' % (k)))

            


def get_default_eval_hparams():
    """default evaluation parameters and settings for a variational model """
    return HParams(num_gpus=2,
                   generate=True,               # flag indicating whether novel images will be generated for the model evaluation
                   reconstruct=True,            # flag indicating whether test images will be reconstructed from the latent codes for the model evaluation
                   compute_logp=True,           # flag indicating whether the marginal log-likelihood by importance will be computed for the model evaluation
                   num_importance_samples=100,  # number of importance samples to be used for the marginal log-likelihood estimation, if compute_logp=True
                   num_sample_images=32,        # number of novel images to be generated if generate=True
                   num_test_images=32,          # number of test images to be reconstructed if reconstruct=True
                   # for the rest of the parameters, check documentation of tf.keras.Model.evaluate.
                   workers=25,
                   batch_size=16,
                   shuffle=True,
                   verbose=True,
                   )

def create_eval_command(args,log_dir):
    """
      Utility function that creates an evaluation command from a training command for a variational model
    """
 
    
    eval_cmnd=['python3']
    
    script=args[0]
    eval_cmnd.append(script)
    
    eval_cmnd.append('--mode=eval')
    
    eval_cmnd.append('--eval_log_dir='+log_dir)
    
    eval_cmnd.append('--model_filepath='+log_dir+'final_model.h5')
    
    train_hparams = [s for s in args if 'train' in s]
    
    for x in train_hparams:
        args.remove(x)
    

    
    eval_cmnd=" ".join([s for s in eval_cmnd+args])
    
    f= open(log_dir+"eval_command.sh","w")
    f.write(eval_cmnd)
    f.close()




