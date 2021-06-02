from latentneural.legacy.lfads.original.run_lfads import main, FLAGS, build_model
from latentneural.lorenz.utils import struct
import numpy as np
import tensorflow as tf
import shutil


def test_lfads_chaotic_rnn_input_pulses_gaussian_noise():
    try:
        shutil.rmtree('./test/legacy/lfads/original/output1')
    except FileNotFoundError as e:
        pass

    tf.compat.v1.reset_default_graph()

    # Run LFADS on chaotic rnn data with no input pulses (g = 1.5) with Gaussian noise
    FLAGS.kind = 'train'
    FLAGS.data_dir = './latentneural/legacy/lfads/original/synth_data/generated/rnn_synth_data_v1.0'
    FLAGS.data_filename_stem = 'gaussian_chaotic_rnn_no_inputs'
    FLAGS.lfads_save_dir = './test/legacy/lfads/original/output1'
    FLAGS.co_dim = 1
    FLAGS.factors_dim = 20
    FLAGS.output_dist = 'gaussian'
    FLAGS.learning_rate_stop = 0.01
    FLAGS.learning_rate_n_to_compare = 1

    main(None)

    shutil.rmtree('./test/legacy/lfads/original/output1')


def test_lfads_inner():
    try:
        shutil.rmtree('./test/legacy/lfads/original/output2')
    except FileNotFoundError as e:
        pass

    tf.compat.v1.reset_default_graph()
    
    datasets = {
        'dummy': {
            'train_data': np.random.randn(200, 100, 10).cumsum(axis=1),
            'valid_data': np.random.randn(50, 100, 10).cumsum(axis=1),
            'data_dim': 10, # Number of neurons
            'num_steps': 100, # Time-steps
            'train_ext_input': None,
            'valid_ext_input': None,
        }
    }

    hps = {
        '_clip_value': 80,
        'batch_size': 20,
        'cell_clip_value': 5.0,
        'cell_weight_scale': 1.0,
        'checkpoint_name': 'lfads_vae',
        'checkpoint_pb_load_name': 'checkpoint',
        'ci_enc_dim': 128,
        'co_dim': 1,
        'co_mean_corr_scale': 0.0,
        'co_prior_var_scale': 0.1,
        'con_dim': 128,
        'controller_input_lag': 1,
        'csv_log': 'fitlog',
        'device': 'gpu:0',
        'do_causal_controller': False,
        'do_feed_factors_to_controller': True,
        'do_reset_learning_rate': False,
        'do_train_encoder_only': False,
        'do_train_io_only': False,
        'do_train_prior_ar_atau': True,
        'do_train_prior_ar_nvar': True,
        'do_train_readin': True,
        'ext_input_dim': 0,
        'factors_dim': 20,
        'feedback_factors_or_rates': 'factors',
        'gen_cell_input_weight_scale': 1.0,
        'gen_cell_rec_weight_scale': 1.0,
        'gen_dim': 200,
        'ic_dim': 64,
        'ic_enc_dim': 128,
        'ic_post_var_min': 0.0001,
        'ic_prior_var_max': 0.1,
        'ic_prior_var_min': 0.1,
        'ic_prior_var_scale': 0.1,
        'inject_ext_input_to_gen': False,
        'keep_prob': 0.95,
        'kind': 'train',
        'kl_co_weight': 1.0,
        'kl_ic_weight': 1.0,
        'kl_increase_steps': 2000,
        'kl_start_step': 0,
        'l2_con_scale': 0.0,
        'l2_gen_scale': 2000.0,
        'l2_increase_steps': 2000,
        'l2_start_step': 0,
        'learning_rate_decay_factor': 0.95,
        'learning_rate_init': 0.01,
        'learning_rate_n_to_compare': 1,
        'learning_rate_stop': 0.01,
        'lfads_save_dir': './test/legacy/lfads/original/output2',
        'max_ckpt_to_keep': 5,
        'max_ckpt_to_keep_lve': 5,
        'max_grad_norm': 200.0,
        'num_steps_for_gen_ic': 9223372036854775807,
        'output_dist': 'gaussian',
        'output_filename_stem': '',
        'prior_ar_atau': 10.0,
        'prior_ar_nvar': 0.1,
        'ps_nexamples_to_process': 9223372036854775807,
        'temporal_spike_jitter_width': 0}

    hps = struct(hps)

    hps.dataset_names = []
    hps.dataset_dims = {}
    for key in datasets:
        hps.dataset_names.append(key)
        hps.dataset_dims[key] = datasets[key]['data_dim']

    # also store down the dimensionality of the data
    # - just pull from one set, required to be same for all sets
    hps.num_steps = list(datasets.values())[0]['num_steps']
    hps.ndatasets = len(hps.dataset_names)

    if hps.num_steps_for_gen_ic > hps.num_steps:
        hps.num_steps_for_gen_ic = hps.num_steps


    config = tf.compat.v1.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)

    sess = tf.compat.v1.Session(config=config)
    with sess.as_default():
        model = build_model(hps, kind="train", datasets=datasets)
        model.train_model(datasets)

    shutil.rmtree('./test/legacy/lfads/original/output2')