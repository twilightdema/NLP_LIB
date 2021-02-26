import tensorflow as tf
import numpy as np

# Functions to build Transfermer model
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  # add extra dimensions to add the padding
  # to the attention logits.
  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model_in, d_model_out, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model_in = d_model_in
    self.d_model_out = d_model_out

    assert d_model_in % self.num_heads == 0

    self.depth = d_model_in // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model_in)
    self.wk = tf.keras.layers.Dense(d_model_in)
    self.wv = tf.keras.layers.Dense(d_model_in)

    self.dense = tf.keras.layers.Dense(d_model_out)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention, 
                                  (batch_size, -1, self.d_model_in))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])

class TransformerEncoder(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(TransformerEncoder, self).__init__()

    self.mha = MultiHeadAttention(d_model, d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, attention_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2, attention_weights

# The modified version of Transformer Encoder layer that reduce output dimension.
# We want to test if it will suit classification task more.
class TransformerBottleNeckEncoder(tf.keras.layers.Layer):
  def __init__(self, d_model_in, d_model_out, num_heads, dff, rate=0.1):
    super(TransformerBottleNeckEncoder, self).__init__()

    self.mha = MultiHeadAttention(d_model_in, d_model_out, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model_out, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, attention_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model_in)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(attn_output)  # (batch_size, input_seq_len, d_model_out)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model_out)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model_out)

    return out2, attention_weights

class TransformerDecoder(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(TransformerDecoder, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training, 
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2

class TransformerEncoderStack(tf.keras.layers.Layer):
  def __init__(self, d_model, num_layer, num_heads, max_len, rate=0.1):
    super(TransformerEncoderStack, self).__init__()
    self.positional_encoding = positional_encoding(max_len, d_model)
    self.encoders = []
    for i in range(num_layer):
        encoder = TransformerEncoder(d_model, num_heads, d_model*2)
        self.encoders.append(encoder)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]
    x = x + self.positional_encoding[:, :seq_len, :]
    attention_weights = []
    for encoder in self.encoders:
        x, attention_weight = encoder(x, training, mask)
        attention_weights.append(attention_weight)
    return x, attention_weights

class TransformerClassifier(tf.keras.Model):
  def __init__(self, d_model, num_layer, num_heads, num_class_out, max_len, rate=0.1):
    super(TransformerClassifier, self).__init__()
    self.transformer = TransformerEncoderStack(d_model, num_layer, num_heads, max_len)
    self.fnn = tf.keras.layers.Dense(num_class_out)

  def call(self, x, training, mask):
    mask = tf.cast(mask, tf.float32)
    out, attention_weights = self.transformer(x, training, mask)
    out = out[:, 0, :] # Pooled output (Gather only position of CLS token)
    out = self.fnn(out)
    return out, attention_weights

# The modified version of Transformer layer that perform different head / d_model in rach layer.
# This is to immitate CNN structure that can extract abstract idea data from input.
class TransformerBottleNeckEncoderStack(tf.keras.layers.Layer):
  def __init__(self, d_model_list, num_heads_list, max_len, rate=0.1):
    super(TransformerBottleNeckEncoderStack, self).__init__()
    self.positional_encoding = positional_encoding(max_len, d_model_list[0])
    self.encoders = []
    for i in range(len(d_model_list) - 1):
        encoder = TransformerBottleNeckEncoder(d_model_list[i], d_model_list[i+1], num_heads_list[i], d_model_list[i+1]*2)
        self.encoders.append(encoder)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]
    x = x + self.positional_encoding[:, :seq_len, :]
    attention_weights = []
    for encoder in self.encoders:
        x, attention_weight = encoder(x, training, mask)
        attention_weights.append(attention_weight)
    return x, attention_weights

# The TransformerBottleNeck with classification layer
class TransformerBottleNeckClassifier(tf.keras.Model):
  def __init__(self, d_model_list, num_heads_list, num_class_out, max_len, rate=0.1):
    super(TransformerBottleNeckClassifier, self).__init__()
    self.transformer = TransformerBottleNeckEncoderStack(d_model_list, num_heads_list, max_len)
    self.fnn = tf.keras.layers.Dense(num_class_out)

  def call(self, x, training, mask):
    mask = tf.cast(mask, tf.float32)
    out, attention_weights = self.transformer(x, training, mask)
    out = out[:, 0, :] # Pooled output (Gather only position of CLS token)
    out = self.fnn(out)
    return out, attention_weights
