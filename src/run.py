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


from model import DeepVAE
from stat_tools import DISTRIBUTION
from util.trainer import keras_train,get_default_train_hparams
from util.eval import keras_eval,get_default_eval_hparams
from util.data import DatasetTransform
from layers import LAYER,DepthWiseAttention

import tensorflow as tf
import tensorflow_datasets as tfds
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


image_info= {"cifar10": {'shape':(32,32,3),'high':255,'low':0, 'binary':False, 'pad_size':0},

             "mnist":{'shape':(28,28,1),'high':255,'low':0, 'binary':True, 'pad_size':2},
             
             "omniglot":{'shape':(28,28,1),'high':1,'low':0, 'binary':True, 'pad_size':2}
             }


def main(_):
    
    ###############################################################################################
    #                                   Parse HyperParameters                                     #
    ###############################################################################################
    
    hparams = DeepVAE.get_default_hparams().update_config(FLAGS.hparams)
    
    transform_hparams = DatasetTransform.get_default_hparams(FLAGS.dataset).update_config(FLAGS.dataset_transform)
    
    dataset_transform= DatasetTransform(transform_hparams,info=image_info[FLAGS.dataset])

    prior_hparams =  DISTRIBUTION[hparams.prior_type].get_default_hparams().update_config(FLAGS.prior_hparams)
 
    posterior_hparams=DISTRIBUTION[hparams.posterior_type].get_default_hparams().update_config(FLAGS.posterior_hparams)
   
    data_dist_hparams=DISTRIBUTION[hparams.data_distribution].get_default_hparams().update_config(FLAGS.data_dist_hparams)

    preproc_encoder_hparams =  DeepVAE.get_default_network_hparams(hparams).preproc_encoder.update_config(FLAGS.preproc_encoder_hparams)
    
    encoder_hparams =  DeepVAE.get_default_network_hparams(hparams).encoder.update_config(FLAGS.encoder_hparams)
    
    merge_encoder_hparams = LAYER[hparams.merge_encoder_type].get_default_hparams().update_config(FLAGS.merge_encoder_hparams)

    decoder_hparams = DeepVAE.get_default_network_hparams(hparams).decoder.update_config(FLAGS.decoder_hparams)
    
    postproc_decoder_hparams = DeepVAE.get_default_network_hparams(hparams).postproc_decoder.update_config(FLAGS.postproc_decoder_hparams)
    
    merge_decoder_hparams = LAYER[hparams.merge_decoder_type].get_default_hparams().update_config(FLAGS.merge_decoder_hparams)

    decoder_attention_hparams = None if not FLAGS.decoder_attention_hparams else DepthWiseAttention.get_default_hparams().update_config(FLAGS.decoder_attention_hparams)
    
    encoder_attention_hparams = None if not FLAGS.encoder_attention_hparams else DepthWiseAttention.get_default_hparams().update_config(FLAGS.encoder_attention_hparams)

    
    ###############################################################################################
    #                                  Create model                                               #
    ###############################################################################################
    
    
    #multi-gpu model.
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = DeepVAE(hparams=hparams,
                        data_dist_hparams=data_dist_hparams,
                        preproc_encoder_hparams=preproc_encoder_hparams,
                        postproc_decoder_hparams=postproc_decoder_hparams,
                        encoder_hparams=encoder_hparams,
                        merge_encoder_hparams=merge_encoder_hparams,
                        decoder_hparams=decoder_hparams,
                        merge_decoder_hparams=merge_decoder_hparams,
                        posterior_hparams=posterior_hparams,
                        prior_hparams=prior_hparams,
                        image_info=image_info[FLAGS.dataset],
                        decoder_attention_hparams=decoder_attention_hparams,
                        encoder_attention_hparams=encoder_attention_hparams,
                        )
    
    ###############################################################################################
    #                                      Execute                                                #
    ###############################################################################################
    
    if FLAGS.mode == "train":

        train_hparams = get_default_train_hparams().update_config(FLAGS.train_hparams)
      
       

    
        keras_train(model,
                    dataset_path=FLAGS.dataset_path,
                    dataset_builder=tfds.builder(FLAGS.dataset) if not FLAGS.dataset_path else None,
                    dataset_transform=dataset_transform,
                    image_info=image_info[FLAGS.dataset],
                    hparams=train_hparams,
                    checkpoint_period=FLAGS.checkpoint_period,
                    train_log_dir=FLAGS.train_log_dir,
                    train_log_subdir=None if not FLAGS.train_log_subdir else FLAGS.train_log_subdir,
                    epoch=FLAGS.epoch,
                    mirrored_strategy= mirrored_strategy
                    )


   

    elif FLAGS.mode == "eval":
           
        eval_hparams = get_default_eval_hparams().update_config(FLAGS.eval_hparams)
        keras_eval (model=model,
                    model_filepath=FLAGS.model_filepath,
                    dataset_builder=tfds.builder(FLAGS.dataset) if not FLAGS.dataset_path else None,
                    dataset_path=FLAGS.dataset_path,
                    dataset_transform=dataset_transform,
                    image_info=image_info[FLAGS.dataset],
                    hparams=eval_hparams,
                    eval_log_dir=FLAGS.eval_log_dir,
                    mirrored_strategy= mirrored_strategy
                    )




if __name__ == "__main__":
    flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Train or evaluate the model.')
    flags.DEFINE_string('dataset', 'cifar10', 'Tensorflow image dataset.')
    flags.DEFINE_string('dataset_path', '', 'Path to user\'s image dataset.')
    flags.DEFINE_string('dataset_transform', '', 'Transformation to be applied on the data.')
    flags.DEFINE_integer('checkpoint_period', 100, 'Number of training epochs every which the model weights will be saved.')
    flags.DEFINE_string('train_log_dir', '', 'Directory for the training logs.')
    flags.DEFINE_string('train_log_subdir', '', 'Sub-directory train_log_dir for the training logs.')
    flags.DEFINE_string('eval_log_dir', '', 'Directory for the evaluation logs.')
    flags.DEFINE_string('model_filepath', '', 'Path to the model weights.')
    flags.DEFINE_string('hparams', "", 'Model hyper-parameters.')
    flags.DEFINE_string('train_hparams', "", 'Training parameters.')
    flags.DEFINE_string('eval_hparams', "", 'Evaluation parameter.')
    flags.DEFINE_string('decoder_hparams', "", 'Hyper-parameters for the decoder layers.')
    flags.DEFINE_string('merge_decoder_hparams', "", 'Hyper-parameters for the cell that merges the latent sample and the context in the decoder.')
    flags.DEFINE_string('encoder_hparams', "", 'Hyper-parameters for the encoder layers.')
    flags.DEFINE_string('merge_encoder_hparams', "", 'Hyper-parameters for the cell that merges the deterministic and stochastic context in the encoder.')
    flags.DEFINE_string('preproc_encoder_hparams', "", 'Hyper-parameters for the pre-processing layer in the encoder.')
    flags.DEFINE_string('postproc_decoder_hparams', "", 'Hyper-parameters for the post-processing layer in the decoder.')
    flags.DEFINE_string('posterior_hparams', "", 'Hyper-parameters for the posterior distribution.')
    flags.DEFINE_string('data_dist_hparams', "", 'Hyper-parameters for the data distribution.')
    flags.DEFINE_string('prior_hparams', "", 'Hyper-parameters for the prior distribution.')
    flags.DEFINE_integer('epoch', 0, 'Epoch from which the training is resumed (if a model_filepath is provided).')
    flags.DEFINE_string('decoder_attention_hparams', "", 'Hyper-parameters for the depth-wise attention in the decoder.')
    flags.DEFINE_string('encoder_attention_hparams', "", 'Hyper-parameters for the depthwise attention in the encoder.')


    tf.compat.v1.app.run(main)



