import numpy as np
import tensorflow as tf
from transformer import TransformerEncoder, TransformerBottleNeckEncoder

# The modified version of Transformer layer that perform different head / d_model in rach layer.
# This is to immitate CNN structure that can extract abstract idea data from input.
class TransformerBottleNeck(tf.keras.layers.Layer):
  def __init__(self, d_model_list, num_heads_list, rate=0.1):
    super(TransformerBottleNeck, self).__init__()
    self.encoders = []
    for i in range(len(d_model_list) - 1):
        encoder = TransformerBottleNeckEncoder(d_model_list[i], d_model_list[i+1], num_heads_list[i], d_model_list[i+1]*2)
        self.encoders.append(encoder)

  def call(self, x, training, mask):
    for encoder in self.encoders:
        x = encoder(x, training, mask)
    return x

# The Transformer with classification layer
class TransformerBottleNeckClassifier(tf.keras.layers.Layer):
  def __init__(self, d_model_list, num_heads_list, num_class_out, rate=0.1):
    super(TransformerBottleNeckClassifier, self).__init__()
    self.transformer = TransformerBottleNeck(d_model_list, num_heads_list)
    self.fnn = tf.keras.layers.Dense(num_class_out)

  def call(self, x, training, mask):
    out = self.transformer(x, training, mask)
    out = out[:, 0, :] # Pooled output (Gather only position of CLS token)
    out = self.fnn(out)
    return out

# Hyper-parameters
D_MODELS = [16, 32, 64, 128]
N_HEADS = [16, 8, 4, 2]

encoder = TransformerBottleNeckClassifier(D_MODELS, N_HEADS, 2)
model = tf.keras.Model()

sample_encoder_layer_output = encoder(
    tf.random.uniform((64, 43, 16)), False, None)

print(sample_encoder_layer_output.shape)  # (batch_size, input_seq_len, d_model)

