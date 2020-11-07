import os
import sys
import numpy as np
import tensorflow as tf

# input of shape [batch, len, d_model]
input_tensor = tf.constant(
  [
    [
      [0.0, 0.1, 0.2,],
      [0.3, 0.4, 0.5,],
      [0.6, 0.7, 0.8,],
      [0.9, 1.0, 1.1,],
    ],
    [
      [0.0, 0.1, 0.2,],
      [0.3, 0.4, 0.5,],
      [0.6, 0.7, 0.8,],
      [0.9, 1.0, 1.1,],
    ],
  ],
  dtype=tf.float32
)

sess = tf.Session()
print('input_tensor')
print(sess.run(input_tensor))
print('input_tensor.shape')
print(input_tensor.get_shape())

BATCH_SIZE = 2
SEQ_LEN = 4
D_MODEL = 3
ATTENTION_HEAD = 2
KEY_SIZE = 5

input_tensor = tf.reshape(input_tensor, [BATCH_SIZE * SEQ_LEN, D_MODEL])
print('Reshape to 2D tensor')
print('input_tensor')
print(sess.run(input_tensor))
print('input_tensor.shape')
print(input_tensor.get_shape())

# Key weight mapping (D_MODEL, KEY_SIZE x HEAD_SIZE)
K1_init = tf.constant_initializer([
  [
    1.0, 1.0, 1.0, 1.0, 1.0, # Head 1
    2.0, 2.0, 2.0, 2.0, 2.0, # Head 2
  ],
  [
    3.0, 3.0, 3.0, 3.0, 3.0,
    4.0, 4.0, 4.0, 4.0, 4.0,
  ],
  [
    5.0, 5.0, 5.0, 5.0, 5.0,
    6.0, 6.0, 6.0, 6.0, 6.0,
  ]
])
K1 = tf.layers.dense(input_tensor, ATTENTION_HEAD * KEY_SIZE, kernel_initializer=K1_init, name='K1')

# K2 represents K weight learnt from another local model
# Note that the head is mismatch from K1
K2_init = tf.constant_initializer([
  [
    2.0, 2.0, 2.0, 2.0, 2.0, # Head 1
    1.0, 1.0, 1.0, 1.0, 1.0, # Head 2
  ],
  [
    4.0, 4.0, 4.0, 4.0, 4.0,
    3.0, 3.0, 3.0, 3.0, 3.0,
  ],
  [
    6.0, 6.0, 6.0, 6.0, 6.0,
    5.0, 5.0, 5.0, 5.0, 5.0,
  ]
])
K2 = tf.layers.dense(input_tensor, ATTENTION_HEAD * KEY_SIZE, kernel_initializer=K2_init, name='K2')

sess.run(tf.global_variables_initializer())

print('K1')
print(sess.run(K1))
print('K1.shape')
print(K1.get_shape())

print('K2')
print(sess.run(K2))
print('K2.shape')
print(K2.get_shape())

# Transpose K,Q,V weight matrix to [ATTENTION_HEAD, D_MODEL, KEY_SIZE]
def transpose_for_matching(K_val):
  # [D_MODEL, ATTENTION_HEAD, KEY_SIZE]
  K_val = np.reshape(K_val, [-1, ATTENTION_HEAD, KEY_SIZE])
  print('K Weight')
  print(K_val)
  print('K Weight shape')
  print(K_val.shape)

  # [ATTENTION_HEAD, D_MODEL, KEY_SIZE]
  K_val = np.transpose(K_val, [1, 0, 2])
  print('K Weight')
  print(K_val)
  print('K Weight shape')
  print(K_val.shape)
  return K_val

# Transpose K,Q,V weight matrix back to [D_MODEL, KEY_SIZE * ATTENTION_HEAD]
def transpose_back_from_matching(K_val):
  # [D_MODEL, ATTENTION_HEAD, KEY_SIZE]
  K_val = np.transpose(K_val, [1, 0, 2])
  print('K Weight')
  print(K_val)
  print('K Weight shape')
  print(K_val.shape)

  # [D_MODEL, ATTENTION_HEAD x KEY_SIZE]
  K_val = np.reshape(K_val, [-1, ATTENTION_HEAD * KEY_SIZE])
  print('K Weight')
  print(K_val)
  print('K Weight shape')
  print(K_val.shape)
  return K_val

with tf.variable_scope('K1', reuse=True):
  K1_val = sess.run(tf.get_variable('kernel'))
with tf.variable_scope('K2', reuse=True):
  K2_val = sess.run(tf.get_variable('kernel'))

print('K1 Weight')
print(K1_val)
print('K1 Weight shape')
print(K1_val.shape)

print('K2 Weight')
print(K2_val)
print('K2 Weight shape')
print(K2_val.shape)

print('Transpose')
K1_val = transpose_for_matching(K1_val)
K2_val = transpose_for_matching(K2_val)

K1_val = transpose_back_from_matching(K1_val)
K2_val = transpose_back_from_matching(K2_val)
