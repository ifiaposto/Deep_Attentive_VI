# Deep Attentive Variational Inference
[Deep Attentive VAE](https://openreview.net/forum?id=T4-65DNlDij)  is the first work that proposes attention mechanisms to build more expressive variational distributions in deep probabilistic models. It achieves state-of-the-art log-likelihoods while using fewer latent layers and requiring less  training time than prior hierarchical VAEs. 

See also:

* [Paper](https://openreview.net/forum?id=T4-65DNlDij)
* [Presentation](https://iclr.cc/media/iclr-2022/Slides/5923_VZ1HWQv.pdf)
* [Poster](https://github.com/ifiaposto/Deep_Attentive_VI/files/8694104/deep_attentive_vae_poster.pdf)
* [Blog Post](https://blog.ml.cmu.edu/2022/05/27/deep-attentive-variational-inference/)



## Requirements

Deep Attentive Vae is built in Python 3.7 using Tensorflow 2.3.0. Use the following command to install the requirements:

```
pip install -r requirements.txt
``` 

## Training Attentive VAE

Below, we provide the training command for several model configurations on public benchmarks. Detailed documentation can be found in src/run.py and src/model/util.py.


</details>
  
</details>

<details><summary>16-layer CIFAR-10 Attentive VAE for 900 epochs   </summary>
  
* Number of trainable Parameters: 118.97M



```shell script
  
python3 run.py --mode=train --dataset=cifar10 --train_log_dir=../cifar10/ --train_log_subdir=16layer_900_epochs \
     --train_hparams=num_gpus=4,batch_size=40,epochs=900,learning_rate=0.01,learning_rate_schedule=cosine_restart_warmup,lr_first_decay_epochs=300 \
     --hparams=layer_latent_shape=[16,16,20],num_layers=16,data_distribution=discretized_logistic_mixture \
     --encoder_attention_hparams=key_dim=20,query_dim=20,use_layer_norm=True \
     --decoder_attention_hparams=key_dim=20,query_dim=20,use_layer_norm=True \
     --decoder_hparams=scale=[0,0],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 \
     --encoder_hparams=scale=[0,0],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 \
     --preproc_encoder_hparams=scale=[0,-2],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 \
     --postproc_decoder_hparams=scale=[2,0],num_blocks=2,num_filters=128 \
    --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,noise_stddev=0.001 \
    --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-1.0,noise_stddev=0.001
```
  
</details>
<details><summary>16-layer CIFAR-10 Attentive VAE for 400 epochs   </summary>
  
* Number of trainable Parameters:  118.97M
  
```shell script
  
    python3 run.py --mode=train --dataset=cifar10 --train_log_dir=../cifar10/ --train_log_subdir=16layer_400_epochs \
     --train_hparams=num_gpus=4,batch_size=40,epochs=400,learning_rate=0.01,learning_rate_schedule=cosine_warmup  \
     --hparams=layer_latent_shape=[16,16,20],num_layers=16,data_distribution=discretized_logistic_mixture \
     --encoder_attention_hparams=key_dim=20,query_dim=20,use_layer_norm=True \
     --decoder_attention_hparams=key_dim=20,query_dim=20,use_layer_norm=True \
     --decoder_hparams=scale=[0,0],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 \
     --encoder_hparams=scale=[0,0],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 \
     --preproc_encoder_hparams=scale=[0,-2],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 \
     --postproc_decoder_hparams=scale=[2,0],num_blocks=2,num_filters=128 \
    --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,noise_stddev=0.001 \
    --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-1.0,noise_stddev=0.001
  
```
 
</details>
 

<details><summary>15-layer OMNIGLOT Attentive VAE for 400 epochs   </summary>
  
* Number of trainable Parameters: 7.87M



```shell script
  
  
  python3 run.py --mode=train --dataset=omniglot --dataset_path=../omniglot_data/chardata.mat --train_log_dir=../omniglot/ --train_log_subdir=15layer_400_epochs \
  --train_hparams=num_gpus=2,batch_size=16,epochs=400,learning_rate=0.01,learning_rate_schedule=cosine_warmup \
  --hparams=layer_latent_shape=[16,16,20],num_layers=15,num_proc_blocks=1,data_distribution=bernoulli \
  --decoder_attention_hparams=key_dim=8,query_dim=8,use_layer_norm=True \
  --encoder_attention_hparams=key_dim=8,query_dim=8,use_layer_norm=False \
  --merge_encoder_hparams=use_nonlocal=True,nonlocop_hparams=[key_dim=8,query_dim=8] \
  --merge_decoder_hparams=use_nonlocal=True,nonlocop_hparams=[key_dim=8,query_dim=8] \
  --decoder_hparams=scale=[0,0],num_blocks=2,num_filters=32 \
  --encoder_hparams=scale=[0,0],num_blocks=2,num_filters=32 \
  --preproc_encoder_hparams=scale=[0,0,-2],num_nodes=2,num_blocks=3,num_filters=32 \
  --postproc_decoder_hparams=scale=[2,0,0],num_nodes=2,use_nonlocal=True,nonlocop_hparams=[key_dim=8,query_dim=8],num_blocks=3,num_filters=32  \
  --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5 \
  --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5

```
  
</details>
 

<details><summary>16-layer CIFAR-10, NVAE for 400 epochs   </summary>
  
* Number of trainable Parameters: 79.21M



```shell script
 
  python3 run.py --mode=train --dataset=cifar10 --train_log_dir=../cifar10/ --train_log_subdir=nvae_400epochs --train_hparams=num_gpus=4,batch_size=40,epochs=400,learning_rate=0.01,learning_rate_schedule=cosine_warmup --hparams=layer_latent_shape=[16,16,20],num_layers=16,data_distribution=discretized_logistic_mixture --decoder_hparams=scale=[0,0],num_blocks=2,num_filters=128 --encoder_hparams=scale=[0,0],num_blocks=2,num_filters=128 --preproc_encoder_hparams=scale=[0,-2],num_blocks=2,num_filters=128 --postproc_decoder_hparams=scale=[2,0],num_blocks=2,num_filters=128 --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5 --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5.0
  
  ```
  
 </details>
 
 The outputs include:
  
  * The learning curves and the performance (in terms of NELBO) on the test data are provided in the 'keras_trainer_log.csv' file.
  
    * val_loss: the total loss of the training objective, including the regularization penalty.
    * val_loss_1: the negative conditional likelihood term of the NELBO.
    * val_loss_2: the KL regularization penalty of the NELBO.
 * The weights of the learned model are in final_model.h5.
 * The command for training the model from scratch can be found in train.sh.
 * The command for evaluating the model can be found in eval_command.sh.

## Evaluating Attentive VAE

The training command outputs the evaluation script in the file eval_command.sh. Some sample evaluation scripts are provided below. Please refer to src/util/eval.py  for detailed documentation.

The evaluation script produces in the designated folder (eval_log_dir) the following files:
   * if hparams.compute_logp=True:  marginal loglikelihood computed by importance sampling.
   * sample_image_x.png: samples images drawn from the model if hparams.generate=True.
   * [original_test_image_x.png,reconstructed_test_image_x.png]: original vs reconstructed image (from the latent codes) if hparams.reconstruct=True.

<details><summary>CIFAR-10 Attentive VAE </summary>

  

```shell script
  
python3 run.py --mode=eval --eval_log_dir=../cifar10/0_16layer_900_epochs/ --model_filepath=../cifar10/0_16layer_900_epochs/final_model.h5 --dataset=cifar10 --eval_hparams=num_gpus=2 --preproc_encoder_hparams=scale=[0,-2],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 --postproc_decoder_hparams=scale=[2,0],num_blocks=2,num_filters=128 --encoder_hparams=scale=[0,0],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 --decoder_hparams=scale=[0,0],use_nonlocal=True,nonlocop_hparams=[key_dim=32,query_dim=32],num_blocks=2,num_filters=128 --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,noise_stddev=0.001 --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-1.0,noise_stddev=0.001 --decoder_attention_hparams=key_dim=20,query_dim=20,use_layer_norm=True --encoder_attention_hparams=key_dim=20,query_dim=20,use_layer_norm=True --hparams=layer_latent_shape=[16,16,20],num_layers=16,data_distribution=discretized_logistic_mixture


```
  
</details>

<details><summary>OMNIGLOT Attentive VAE </summary>


```shell script

python3 run.py --mode=eval --eval_log_dir=../omniglot/0_15layer_400_epochs/ --model_filepath=../omniglot/0_15layer_400_epochs/final_model.h5 --dataset=omniglot --dataset_path=../omniglot_data/chardata.mat --preproc_encoder_hparams=scale=[0,0,-2],num_nodes=2,num_blocks=3,num_filters=32 --postproc_decoder_hparams=scale=[2,0,0],num_nodes=2,use_nonlocal=True,nonlocop_hparams=[key_dim=8,query_dim=8],num_blocks=3,num_filters=32 --merge_encoder_hparams=use_nonlocal=True,nonlocop_hparams=[key_dim=8,query_dim=8] --encoder_hparams=scale=[0,0],num_blocks=2,num_filters=32 --merge_decoder_hparams=use_nonlocal=True,nonlocop_hparams=[key_dim=8,query_dim=8] --decoder_hparams=scale=[0,0],num_blocks=2,num_filters=32 --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5 --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5 --decoder_attention_hparams=key_dim=8,query_dim=8,use_layer_norm=True --encoder_attention_hparams=key_dim=8,query_dim=8,use_layer_norm=False --hparams=layer_latent_shape=[16,16,20],num_layers=15,num_proc_blocks=1,data_distribution=bernoulli
                                                                                              

```
</details>
  
## Saved Models 

Saved models, learning curves, and qualitative evaluations of the models can be found below:
 * [CIFAR-10](https://drive.google.com/file/d/1AQrc6Bx7ktLThR7GrcHzRti2VyV7hSI3/view?usp=sharing)
 * [OMNIGLOT]( https://drive.google.com/file/d/1SSWGvQd1pTgUieSpQe7YPcfoB5XHEquc/view?usp=sharing)

## Datasets

* The omniglot was partitioned and preprocessed as in [here](https://github.com/yburda/iwae/tree/master/datasets/OMNIGLOT), and it can also be downloaded [here](https://drive.google.com/file/d/1yceW1Lbt3oSSZx_Fy6PMZ_8LWjpRyEtV/view?usp=sharing). 
* The rest of the datasets can be downloaded through [Tensorflow Datasets](https://www.tensorflow.org/datasets).

## BibTex
Please cite our paper, if you happen to use this codebase:


```

@inproceedings{apostolopoulou2021deep,
  title={Deep Attentive Variational Inference},
  author={Apostolopoulou, Ifigeneia and Char, Ian and Rosenfeld, Elan and Dubrawski, Artur},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

## Funding Information

Funding for the development of this library has been generously provided by the following sponsors:

| **Onassis Graduate Fellowship**  | **Leventis Graduate Fellowship**  |**DARPA**|**NSF**|
| --------------- | --------------- |---------------|---------------|
| awarded to Ifigeneia  Apostolopoulou    | awarded to Ifigeneia Apostolopoulou     |AWARD FA8750-17-2-0130    | AWARD  2038612 & <br> Graduate Research <br> Fellowship  awarded <br> to  Ian Char|
| <img src="https://user-images.githubusercontent.com/11561732/168452919-ee649b21-7f88-4e50-b9be-e17c80870de0.png" alt="onassis_logo" width="200px" height="150px">   | <img src="https://user-images.githubusercontent.com/11561732/168452909-64ff573c-19a8-4db4-b708-381765cd4313.jpg"  width="150px" height="150px">|<img src="https://user-images.githubusercontent.com/11561732/168449987-62391aa4-77ff-491e-a9f9-89a5b2ae7a56.jpg"  width="200px" height="150px"> |<img src="https://user-images.githubusercontent.com/11561732/168450030-68fd8baa-c3e5-4ca0-ba75-bd3e783df56b.png"  width="200px" height="150px">
