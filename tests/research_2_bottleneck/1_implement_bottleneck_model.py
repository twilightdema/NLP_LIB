import numpy as np
import tensorflow as tf
from transformer import TransformerEncoder

# Hyper-parameters
D_MODELS = [32, 32, 32, 32]
N_HEADS = [2, 4, 8, 16]

sample_encoder_layer = TransformerEncoder(512, 8, 2048)

sample_encoder_layer_output = sample_encoder_layer(
    tf.random.uniform((64, 43, 512)), False, None)

print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

