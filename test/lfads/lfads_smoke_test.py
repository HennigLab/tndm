from lfads.run_lfads import main, FLAGS


def test_lfads_chaotic_rnn_input_pulses_gaussian_noise():
    # Run LFADS on chaotic rnn data with no input pulses (g = 1.5) with Gaussian noise
    FLAGS.kind = 'train'
    FLAGS.data_dir = './lfads/synth_data/generated/rnn_synth_data_v1.0'
    FLAGS.data_filename_stem = 'gaussian_chaotic_rnn_no_inputs'
    FLAGS.lfads_save_dir = './lfads/synth_data/output/lfads_chaotic_rnn_inputs_g2p5'
    FLAGS.co_dim = 1
    FLAGS.factors_dim = 20
    FLAGS.output_dist = 'gaussian'
    FLAGS.learning_rate_stop = 0.01
    FLAGS.learning_rate_n_to_compare = 1

    main(None)