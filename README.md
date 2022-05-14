# Deep Attentive Variationa Inference
Author's implementation of the paper: "Deep Attentive Variational Inference", ICLR 2022


## Requirements

NVAE is built in Python 3.7 using Tensorflow 2.3.0. Use the following command to install the requirements:

```
pip install -r requirements.txt
``` 

## Training NVAE

Below, we provide the training command for several model configurations on CIFAR-10.
You may refer to cifar10.sh for a larger list.
In run_vae.py, we provide detailed documentation on the training command.

</details>

<details><summary>Vanilla VAE (single layer) </summary>

  
  
* Number of trainable Parameters: 2.213556M
* NELBO after 400 training epochs: 8405.23 (3.95 bits per dimension)
* Marginal Likelihood (with 100 importance samples): -8318.75 (3.91 bits per dimension)

```shell script
  
 python run_vae.py --mode=train --train_log_dir=cifar10/ --train_log_subdir=vanilla_20chan_batchnorm  --train_hparams=use_mirrored_strategy=True,num_gpus=4,optimizer=adamax,optimization_method=gradient_descent,kl_scheduler=[warmup_smooth_start=1.0,warmup_epochs=0,update_beta=smooth],batch_size=32,epochs=400,learning_rate=0.01,learning_rate_schedule=cosine_warmup,lr_warmup_epochs=6,lr_decay_epochs=394,lr_steps_per_epoch=1563,lr_min_learning_rate=0.0001 --eval_hparams=num_importance_samples=100,workers=25,compute_logp=True --hparams=layer_latent_shape=[16,16,20],nof_layers=1,data_distribution=discretized_logistic_mixture,posterior_type=gaussian_diag,prior_type=gaussian_diag,lambda_reg=1.5e-4,norm_reg=l2,residual_var_layer=True --preproc_encoder_hparams=scale=[0,-2],use_stem=True,activation=swish,use_weight_norm=True,use_batch_norm=True,bn_momentum=0.95,nof_blocks=2,nof_filters=128,use_data_init=False --postproc_decoder_hparams=scale=[2,0],use_stem=True,stem_kernel_size=1,activation=swish,use_weight_norm=True,use_batch_norm=True,bn_momentum=0.95,nof_blocks=2,nof_filters=256,use_data_init=False --data_dist_hparams=network_hparams=[kernel_size=3,num_times=1,activation=elu] --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,network_hparams=[kernel_size=3,num_times=2,activation=elu]


```
  
</details>

<details><summary>8-layer NVAE  </summary>
  
* Number of trainable Parameters: 39.44066M
* NELBO after 400 training epochs: 6622.57 ( 3.11 bits per dimension)
* Marginal Likelihood (with 100 importance samples): -6556.962 ( 3.08 bits per dimension)


```shell script
  
python run_vae.py --mode=train --train_log_dir=cifar10/ --train_log_subdir=8layer_20chan_nobatchnorm  --train_hparams=use_mirrored_strategy=True,num_gpus=4,optimizer=adamax,optimization_method=gradient_descent,kl_scheduler=[warmup_smooth_start=1.0,warmup_epochs=0,update_beta=smooth],batch_size=32,epochs=400,learning_rate=0.01,learning_rate_schedule=cosine_warmup,lr_warmup_epochs=5,lr_decay_epochs=394,lr_steps_per_epoch=391,lr_min_learning_rate=0.0001 --eval_hparams=num_importance_samples=100,workers=25,batch_size=32,compute_logp=True,generate=True,reconstruct=True,verbose=True --hparams=layer_latent_shape=[16,16,20],nof_layers=8,data_distribution=discretized_logistic_mixture,posterior_type=gaussian_diag,prior_type=gaussian_diag,lambda_reg=1.15e-4,norm_reg=l2,residual_var_layer=True --merge_encoder_hparams=use_weight_norm=True,use_data_init=False --merge_generator_hparams=use_weight_norm=True,use_data_init=False --generator_hparams=scale=[0,0],use_stem=False,activation=swish,use_weight_norm=True,use_batch_norm=False,nof_blocks=2,nof_filters=256,use_data_init=False --encoder_hparams=scale=[0,0],use_stem=False,activation=swish,use_weight_norm=True,use_batch_norm=False,nof_blocks=2,nof_filters=256,use_data_init=False --preproc_encoder_hparams=scale=[0,-2],use_stem=True,activation=swish,use_weight_norm=True,use_batch_norm=False,nof_blocks=2,nof_filters=128,use_data_init=False --postproc_decoder_hparams=scale=[2,0],use_stem=False,activation=swish,use_weight_norm=True,use_batch_norm=False,nof_blocks=2,nof_filters=128,use_data_init=False --data_dist_hparams=network_hparams=[kernel_size=3,num_times=1,activation=elu] --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,network_hparams=[kernel_size=3,num_times=1,activation=] --init_posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,network_hparams=[kernel_size=3,num_times=2,activation=elu] --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,network_hparams=[kernel_size=1,num_times=1,activation=elu]

```
  
</details>

<details><summary>16-layer NVAE with batch normalization and Squeeze and Excite blocks  </summary>
  
* Number of trainable Parameters:  79.746204M
* NELBO after 400 training epochs: 6402.88 ( 3.01 bits per dimension)
* Marginal Likelihood (with 100 importance samples): -6334.46 ( 2.97 bits per dimension)
  
```shell script
  
  python run_vae.py  --mode=train --train_log_dir=cifar10/ --train_log_subdir=16layer_20chan_batchnorm_se  --train_hparams=use_mirrored_strategy=True,num_gpus=4,optimizer=adamax,optimization_method=gradient_descent,kl_scheduler=[warmup_smooth_start=0.1,warmup_epochs=16,update_beta=smooth],batch_size=32,epochs=400,learning_rate=0.01,learning_rate_schedule=cosine_warmup,lr_warmup_epochs=5,lr_decay_epochs=394,lr_steps_per_epoch=391,lr_min_learning_rate=0.0001 --eval_hparams=num_importance_samples=100,workers=25,batch_size=32,compute_logp=True,generate=True,reconstruct=True,verbose=True --hparams=layer_latent_shape=[16,16,20],nof_layers=16,lambda_reg=1.15e-4,norm_reg=l2,residual_var_layer=True,generator_type=resnet --merge_encoder_hparams=use_weight_norm=True,use_data_init=False --merge_generator_hparams=use_weight_norm=True,use_data_init=False --generator_hparams=scale=[0,0],use_se=True,use_stem=False,activation=swish,use_weight_norm=True,use_batch_norm=True,bn_momentum=0.95,nof_blocks=2,nof_filters=256,use_data_init=False --encoder_hparams=scale=[0,0],use_se=True,use_stem=False,activation=swish,use_weight_norm=True,use_batch_norm=True,bn_momentum=0.95,nof_blocks=2,nof_filters=256,use_data_init=False --preproc_encoder_hparams=scale=[0,-2],use_se=True,use_stem=True,activation=swish,use_weight_norm=True,use_batch_norm=True,bn_momentum=0.95,nof_blocks=2,nof_filters=128,use_data_init=False --postproc_decoder_hparams=scale=[2,0],use_stem=False,use_se=True,activation=swish,use_weight_norm=True,use_batch_norm=True,bn_momentum=0.95,nof_blocks=2,nof_filters=128,use_data_init=False --data_dist_hparams=network_hparams=[kernel_size=3,num_times=1,activation=elu] --posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,network_hparams=[kernel_size=3,num_times=1,activation=] --init_posterior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,network_hparams=[kernel_size=3,num_times=2,activation=elu] --prior_hparams=log_scale_upper_bound=5,log_scale_low_bound=-5,network_hparams=[kernel_size=1,num_times=1,activation=elu]


```
 </details>
  
  The learning curves and the performance (in terms of NELBO) on the test data are provided in the 'keras_trainer_log.csv' file.
  
  * val_loss: the total loss of the training objective, including the regularization penalty.
  * val_loss_1: the negative conditional likelihood term of the NELBO.
  * val_loss_2: the KL regularization penalty of the NELBO.



## Funding Information

Funding for the development of this library has been generously provided by the following sponsors:

| **Onassis Graduate Fellowship**  | **Leventis Graduate Fellowship**  |**DARPA**|**NSF**|
| --------------- | --------------- |---------------|---------------|
| awarded to Ifigeneia  Apostolopoulou    | awarded to Ifigeneia Apostolopoulou     |AWARD FA8750-17-2-0130    | AWARD  2038612 & <br> Graduate Research <br> Fellowship  awarded <br> to  Ian Char|
| <img src="https://github.com/ifiaposto/Tensorflow-Implementation-of-NVAE/blob/main/img/onassis_logo.png" alt="onassis_logo" width="200px" height="150px">   | <img src="https://github.com/ifiaposto/Tensorflow-Implementation-of-NVAE/blob/main/img/leventis_logo.jpg"  width="150px" height="150px">   |<img src="https://user-images.githubusercontent.com/11561732/168449987-62391aa4-77ff-491e-a9f9-89a5b2ae7a56.jpg"  width="200px" height="150px"> |<img src="https://user-images.githubusercontent.com/11561732/168450030-68fd8baa-c3e5-4ca0-ba75-bd3e783df56b.png"  width="200px" height="150px">
