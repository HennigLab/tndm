#!/bin/bash

# Copyright 2017 Google Inc. All Rights Reserved.
#
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
#
# ==============================================================================

DATA_DIR=latentneural/data/storage/lorenz/20210610T215300
DATA_STEM=dataset.h5

co_dim=0
gen_dim=64
ic_dim=64
ic_enc_dim=64
batch_size=8 #was 10
batch_size_eval=128
learning_rate_init=.01
learning_rate_stop=1e-05
learning_rate_decay_factor=.95
learning_rate_n_to_compare=6
keep_prob=.95 #was .95 
l2_gen_scale=2000.0
kl_ic_weight=1.0
kl_increase_steps=2000
kl_start_step=0
seed=0
factors_dim=3

RESULTS_DIR=latentneural/data/storage/lorenz/20210610T215300/results

echo "Running LFADS on data"
python latentneural/legacy/lfads/original/run_lfads.py --kind=train --data_dir=$DATA_DIR --data_filename_stem=$DATA_STEM --lfads_save_dir=$RESULTS_DIR --co_dim=$co_dim --factors_dim=$factors_dim --ext_input_dim=0 --controller_input_lag=1 --output_dist=poisson --do_causal_controller=false --batch_size=$batch_size --learning_rate_init=$learning_rate_init --learning_rate_stop=$learning_rate_stop --learning_rate_decay_factor=$learning_rate_decay_factor --learning_rate_n_to_compare=$learning_rate_n_to_compare --do_reset_learning_rate=false --keep_prob=$keep_prob --gen_dim=$gen_dim --ci_enc_dim=128 --ic_dim=$ic_dim --ic_enc_dim=$ic_enc_dim --ic_prior_var_min=0.1 --gen_cell_input_weight_scale=1.0 --cell_weight_scale=1.0 --do_feed_factors_to_controller=true --kl_start_step=$kl_start_step --kl_increase_steps=$kl_increase_steps --kl_ic_weight=$kl_ic_weight --l2_gen_scale=$l2_gen_scale --l2_con_scale=0.0 --l2_start_step=0 --l2_increase_steps=2000 --ic_prior_var_scale=0.1 --ic_post_var_min=0.0001 --kl_co_weight=1.0 --prior_ar_nvar=0.1 --cell_clip_value=5.0 --max_ckpt_to_keep_lve=5 --do_train_prior_ar_atau=true --co_prior_var_scale=0.1 --csv_log=fitlog --feedback_factors_or_rates=factors --do_train_prior_ar_nvar=true --max_grad_norm=200.0 --device=cpu:0 --num_steps_for_gen_ic=100000000 --ps_nexamples_to_process=100000000 --checkpoint_name=lfads_vae --temporal_spike_jitter_width=0 --checkpoint_pb_load_name=checkpoint --inject_ext_input_to_gen=false --co_mean_corr_scale=0.0 --gen_cell_rec_weight_scale=1.0 --max_ckpt_to_keep=5 --output_filename_stem="" --ic_prior_var_max=0.1 --prior_ar_atau=10.0 --do_train_io_only=false --do_train_encoder_only=false --seed=$seed

echo "Evaluating LFADS on data"
python latentneural/legacy/lfads/original/run_lfads.py --kind=posterior_sample_and_average --data_dir=$DATA_DIR --data_filename_stem=$DATA_STEM --lfads_save_dir=$RESULTS_DIR --co_dim=$co_dim --factors_dim=$factors_dim --ext_input_dim=0 --controller_input_lag=1 --output_dist=poisson --do_causal_controller=false --batch_size=$batch_size_eval --learning_rate_init=$learning_rate_init --learning_rate_stop=$learning_rate_stop --learning_rate_decay_factor=$learning_rate_decay_factor --learning_rate_n_to_compare=$learning_rate_n_to_compare --do_reset_learning_rate=false --keep_prob=$keep_prob --gen_dim=$gen_dim --ci_enc_dim=128 --ic_dim=$ic_dim --ic_enc_dim=$ic_enc_dim --ic_prior_var_min=0.1 --gen_cell_input_weight_scale=1.0 --cell_weight_scale=1.0 --do_feed_factors_to_controller=true --kl_start_step=$kl_start_step --kl_increase_steps=$kl_increase_steps --kl_ic_weight=$kl_ic_weight --l2_gen_scale=$l2_gen_scale --l2_con_scale=0.0 --l2_start_step=0 --l2_increase_steps=2000 --ic_prior_var_scale=0.1 --ic_post_var_min=0.0001 --kl_co_weight=1.0 --prior_ar_nvar=0.1 --cell_clip_value=5.0 --max_ckpt_to_keep_lve=5 --do_train_prior_ar_atau=true --co_prior_var_scale=0.1 --csv_log=fitlog --feedback_factors_or_rates=factors --do_train_prior_ar_nvar=true --max_grad_norm=200.0 --device=cpu:0 --num_steps_for_gen_ic=100000000 --ps_nexamples_to_process=100000000 --checkpoint_name=lfads_vae --temporal_spike_jitter_width=0 --checkpoint_pb_load_name=checkpoint --inject_ext_input_to_gen=false --co_mean_corr_scale=0.0 --gen_cell_rec_weight_scale=1.0 --max_ckpt_to_keep=5 --output_filename_stem="" --ic_prior_var_max=0.1 --prior_ar_atau=10.0 --do_train_io_only=false --do_train_encoder_only=false --seed=$seed
