import tensorflow as tf
from typing import Dict, Any

from latentneural.utils import ArgsParser
from .sampling import Sampling
from .crop import Crop


class TNDM(tf.keras.Model):

  def __init__(self, **kwargs: Dict[str, Any]):
    super(TNDM, self).__init__()

    relevant_dynamics: int = ArgsParser.get_or_default(kwargs, 'relevant_dynamics', 64)
    irrelevant_dynamics: int = ArgsParser.get_or_default(kwargs, 'irrelevant_dynamics', 64)
    relevant_factors: int = ArgsParser.get_or_default(kwargs, 'relevant_factors', 2)
    irrelevant_factors: int = ArgsParser.get_or_default(kwargs, 'irrelevant_factors', 1)
    behavioural_space: int = ArgsParser.get_or_default(kwargs, 'behaviour_space', 1)
    neural_space: int = ArgsParser.get_or_default(kwargs, 'neural_space', 50)

    # ENCODER
    encoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'encoder', {})
    self.encoder = tf.keras.layers.GRU(max([relevant_dynamics, irrelevant_dynamics]), time_major=False, name="EncoderRNN", **encoder_args)
    dense_relevant_mean_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_relevant_mean', {})
    self.dense_relevant_mean = tf.keras.layers.Dense(relevant_dynamics, name="DenseRelevantMean", **dense_relevant_mean_args)
    dense_relevant_logvar_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_relevant_logvar', {})
    self.dense_relevant_logvar = tf.keras.layers.Dense(relevant_dynamics, name="DenseRelevantLogVar", **dense_relevant_logvar_args)
    dense_irrelevant_mean_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_irrelevant_mean', {})
    self.dense_irrelevant_mean = tf.keras.layers.Dense(relevant_dynamics, name="DenseIrrelevantMean", **dense_irrelevant_mean_args)
    dense_irrelevant_logvar_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'dense_irrelevant_logvar', {})
    self.dense_irrelevant_logvar = tf.keras.layers.Dense(relevant_dynamics, name="DenseIrrelevantLogVar", **dense_irrelevant_logvar_args)

    # SAMPLING
    self.relevant_sampling = Sampling(name="RelevantSampling")
    self.irrelevant_sampling = Sampling(name="IrrelevantSampling")
    
    # DECODERS
    relevant_decoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'relevant_decoder', {})
    self.relevant_decoder = tf.keras.layers.GRU(relevant_dynamics, return_sequences=True, time_major=False, name="RelevantRNN", **relevant_decoder_args)
    irrelevant_decoder_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'irrelevant_decoder', {})
    self.irrelevant_decoder = tf.keras.layers.GRU(irrelevant_dynamics, return_sequences=True, time_major=False, name="IrrelevantRNN", **irrelevant_decoder_args)

    # DIMENSIONALITY REDUCTION
    relevant_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'relevant_dense', {})
    self.relevant_dense = tf.keras.layers.Dense(relevant_factors, name="RelevantDense", **relevant_dense_args)
    irrelevant_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'irrelevant_dense', {})
    self.irrelevant_dense = tf.keras.layers.Dense(irrelevant_factors, name="IrrelevantDense", **irrelevant_dense_args)

    # BEHAVIOUR
    behavioural_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'behavioural_dense', {})
    self.behavioural_dense = tf.keras.layers.Dense(behavioural_space, name="BehaviouralDense", **behavioural_dense_args)

    # NEURAL
    self.neural_concatenation = tf.keras.layers.Concatenate(name="NeuralConcat")
    neural_dense_args: Dict[str, Any] = ArgsParser.get_or_default(kwargs, 'neural_dense', {})
    self.neural_dense = tf.keras.layers.Dense(neural_space, name="NeuralDense", **neural_dense_args)
    

  def call(self, inputs, training: bool=False):
    encoded = self.encoder(inputs, training=training)

    r_mean = self.dense_relevant_mean(encoded)
    r_logvar = self.dense_relevant_logvar(encoded)
    i_mean = self.dense_irrelevant_mean(encoded)
    i_logvar = self.dense_irrelevant_logvar(encoded)

    g0_r = self.relevant_sampling(tf.stack([r_mean, r_logvar], axis=-1), training=training)
    g0_i = self.irrelevant_sampling(tf.stack([i_mean, i_logvar], axis=-1), training=training)
    
    # Assuming inputs are zero and everything comes from the GRU
    u_r = tf.stack([tf.zeros_like(inputs)[:,:,-1] for i in range(self.relevant_decoder.units)], axis=-1)
    u_i = tf.stack([tf.zeros_like(inputs)[:,:,-1] for i in range(self.irrelevant_decoder.units)], axis=-1)
    
    g_r = self.relevant_decoder(u_r, initial_state=g0_r, training=training)
    g_i = self.irrelevant_decoder(u_i, initial_state=g0_i, training=training)

    z_r = self.relevant_dense(g_r, training=training)
    z_i = self.irrelevant_dense(g_i, training=training)

    b = self.behavioural_dense(z_r)

    z = self.neural_concatenation([z_r, z_i])
    f = self.neural_dense(z)

     # In order to be able to auto-encode, the dimensions should be the same
    if not self.built:
      assert all([f_i == i_i for f_i, i_i in zip(list(f.shape), list(inputs.shape))])

    return b, f, (r_mean, r_logvar), (i_mean, i_logvar)