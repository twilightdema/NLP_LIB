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

# Query weight mapping (D_MODEL, KEY_SIZE x HEAD_SIZE)
Q1_init = tf.constant_initializer([
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
Q1 = tf.layers.dense(input_tensor, ATTENTION_HEAD * KEY_SIZE, kernel_initializer=Q1_init, name='Q1')

# Q2 represents Q weight learnt from another local model
# Note that the head is mismatch from Q1
Q2_init = tf.constant_initializer([
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
Q2 = tf.layers.dense(input_tensor, ATTENTION_HEAD * KEY_SIZE, kernel_initializer=Q2_init, name='Q2')

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
  print('Weight')
  print(K_val)
  print('Weight shape')
  print(K_val.shape)

  # [ATTENTION_HEAD, D_MODEL, KEY_SIZE]
  K_val = np.transpose(K_val, [1, 0, 2])
  print('Weight (reshaped)')
  print(K_val)
  print('Weight shape (reshaped)')
  print(K_val.shape)
  return K_val

# Transpose K,Q,V weight matrix back to [D_MODEL, KEY_SIZE * ATTENTION_HEAD]
def transpose_back_from_matching(K_val):
  # [D_MODEL, ATTENTION_HEAD, KEY_SIZE]
  K_val = np.transpose(K_val, [1, 0, 2])
  print('Weight')
  print(K_val)
  print('Weight shape')
  print(K_val.shape)

  # [D_MODEL, ATTENTION_HEAD x KEY_SIZE]
  K_val = np.reshape(K_val, [-1, ATTENTION_HEAD * KEY_SIZE])
  print('Weight (reshaped)')
  print(K_val)
  print('Weight shape (reshaped)')
  print(K_val.shape)
  return K_val

def generate_permutaion_matrix(perm_size):
  current = [i for i in range(perm_size)]
  def gen_perm(pos):
    if pos == len(current):
      yield current
    else:
      for i in range(pos, len(current)):
        src = current[pos]
        tgt = current[i]
        current[i] = src
        current[pos] = tgt
        yield from gen_perm(pos + 1)
        current[pos] = src
        current[i] = tgt
  yield from gen_perm(0)

def apply_permutation_matrix(perm_set, perm_mat):
  return np.array([[input_list[i] for i in perm_mat] for input_list in perm_set])

def distance_function(list1, list2):
  acc_dist = np.sum(np.abs(list1 - list2))
  print('Distance = ' + str(acc_dist))
  return acc_dist

def find_best_permutation_matrix(list1, list2):
  list1 = np.array(list1)
  list2 = np.array(list2)
  head_count = list1[0].shape[0]
  perm_mats = generate_permutaion_matrix(head_count)
  min_distance = 1.0e+6
  min_perm_mat = None
  for perm_mat in perm_mats:
    permutated_w1 = apply_permutation_matrix(list1, perm_mat)
    print(' - Matching: ' + str(permutated_w1) + ' with ' + str(list2) + ' (perm_mat = ' + str(perm_mat) + ')')
    distance = distance_function(permutated_w1, list2)
    if distance < min_distance:
      min_distance = distance
      min_perm_mat = list(np.array(perm_mat))
  return min_perm_mat, min_distance

def calculate_federated_weights(weight_list):
  return np.average(np.array(weight_list), axis=0)

with tf.variable_scope('K1', reuse=True):
  K1_val = sess.run(tf.get_variable('kernel'))
  K1_bias = sess.run(tf.get_variable('bias'))
with tf.variable_scope('K2', reuse=True):
  K2_val = sess.run(tf.get_variable('kernel'))
  K2_bias = sess.run(tf.get_variable('bias'))

with tf.variable_scope('Q1', reuse=True):
  Q1_val = sess.run(tf.get_variable('kernel'))
with tf.variable_scope('Q2', reuse=True):
  Q2_val = sess.run(tf.get_variable('kernel'))

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

K1_bias = transpose_for_matching(K1_bias)
K2_bias = transpose_for_matching(K2_bias)

print('K1 Bias')
print(K1_bias)
print('K1 Bias shape')
print(K1_bias.shape)

Q1_val = transpose_for_matching(Q1_val)
Q2_val = transpose_for_matching(Q2_val)

perm_set_1 = [K1_val, Q1_val, K1_bias]
perm_set_2 = [K2_val, Q2_val, K2_bias]

min_perm_mat, min_distance = find_best_permutation_matrix(perm_set_1, perm_set_2)

print('Minimum Distance = ' + str(min_distance))
print('ARGMIN Permutation Matrix = ' + str(min_perm_mat))

federated_weights = calculate_federated_weights([perm_set_1, 
  perm_set_2]
)
print('Normal Federated Weights = ' + str(federated_weights))

federated_weights = calculate_federated_weights([apply_permutation_matrix(perm_set_1, min_perm_mat), 
  perm_set_2]
)
print('Matched Federated Weights = ' + str(federated_weights))

K_val = transpose_back_from_matching(federated_weights[0])
Q_val = transpose_back_from_matching(federated_weights[1])
