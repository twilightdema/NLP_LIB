# This experiment is for comparing case of using matched head algorithm,
# comparing with vanilla FedAVG.
# For benchmarking, we perform 100 rounds of experiments and do analysis to see proportion of Good / Bad results
# so that we can find the way to improve the chance of "Good" result by looking at distance function.
# For very basic case, we want to test if head matching really help attention behavior in Multi-Head Attention.
# To cut other factor out, we strictly test only attention mechanism as below:
# 1) Input Embedding is fixed as 'augmented' one-hot encoding and not trainable.
# 2) Positional Encoding is fixed and not trainable.
# 3) We instrument the network by puttting label as direct 'Attention Output' as in 4)
# 4) The objective of the model is to direct attention to specefic token with the conditions below:
#    - input token has CLS, SEP, a, b, c, d, e, f, g, h (CLS is always the 1st position)
#    - if there is 'd' after 'c', output will set to 1.0 at the latest position of 'c' before 'd'.
#    - if there is 'e' before 'f', output will set to 1.0 at the first position of 'f' after 'e'.
#    - if there is any of above two cases active, output at cls position is set to 1.0.
#
#  Example:     CLS g h f e d c c d e g h SEP
#  Label:        1  0 0 0 0 0 0 1 0 0 0 0  0
#  Example:     CLS g h f e d c c c f e h SEP
#  Label:        1  0 0 0 0 0 0 0 0 1 0 0  0
#  Example:     CLS g h f e d c c c e g h SEP
#  Label:        0  0 0 0 0 0 0 0 0 0 0 0  0
#
import os
import sys
import requests
import zipfile
import pickle
import numpy as np
import sentencepiece as spm
import tensorflow as tf
import math
import random
import json

# Benchmark parameters
TRIAL_NUM = 100
current_trial_round = 0

# Model configuration
USE_POSITIONAL_ENCODING = True

# Algorithm of weight matching to be used
MATCH_USING_EUCLIDIAN_DISTANCE = False
MATCH_USING_COSINE_SIMILARITY = True

# Training Parameters
COMMUNICATION_ROUNDS = 8
LOCAL_TRAIN_EPOCH = 100
ATTENTION_HEAD = 4
BATCH_SIZE = 5
BATCH_NUM = 10
D_MODEL = 48
SEQ_LEN = 10
VOCAB_SIZE = 150

# Number of federated nodes
NODE_COUNT = 2

# String speical token specifications
TOKEN_UNKNOWN = 1
TOKEN_CLS = 2
TOKEN_SEP = 3

############################################################
# FUNCTIONS FOR SIMULATE TRAINING DATA
dict_vocab = {
  0: '',
  1: '[UNK]',
  2: '[CLS]',
  3: '[SEP]',
  4: 'a',
  5: 'b',
  6: 'c',
  7: 'd',
  8: 'e',
  9: 'f',
  10: 'g',
  11: 'h',
}

vocab_dict = { dict_vocab[id]: id for id in dict_vocab }

# For decode one-hot input format back to text token
oh_input_map = {}

def decode_input_ids(input_ids):
  ret = []
  input_ids = np.array(input_ids)
  input_toks = np.argmax(input_ids, axis=-1)
  for tok in input_toks:
    ret.append(dict_vocab[tok])
  return ret

def decode_input_oh_batch(input_batch):
  ret = []
  for inputs in input_batch:
    tokens = []
    for ii_oh in inputs:
      tokens.append(oh_input_map[json.dumps(ii_oh)])
    ret.append(tokens)
  return ret

def simulate_output(input_seq):
  case_1 = False 
  case_2 = False
  case_1_idx = -1
  case_2_idx = -1

  for idx, id in enumerate(input_seq):
    if dict_vocab[id] == 'c' and not case_1:
      case_1_idx = idx
    if dict_vocab[id] == 'd' and case_1_idx != -1:
      case_1 = True
    if dict_vocab[id] == 'e' and not case_2:
      case_2 = True
    if dict_vocab[id] == 'f' and case_2 and case_2_idx == -1:
      case_2_idx = idx
  if not case_1:
    case_1_idx = -1
  if case_2_idx == -1:
    case_2 = False

  label = [0.0 for _ in range(len(input_seq))]
  if case_1 or case_2:
    label[0] = 1.0
  if case_1_idx != -1:
    label[case_1_idx] = 1.0
  if case_2_idx != -1:
    label[case_2_idx] = 1.0
  return label

def generate_random(smallest, largest, mean, num):
  print('mean = ' + str(mean))
  arr = np.random.randint(smallest, largest, num)
  print('1->' + str(arr))
  i = np.sum(arr) - mean * num
  chunk = int(num / 2) # Delete mean excess from half of the array
  reduc = i / chunk 
  decm, intg = math.modf(reduc)
  args = np.argsort(arr)
  if reduc < 0.0:
    args = args[::-1]
  arr[args[-chunk-1:-1]] -= int(intg)
  arr[args[-1]] -= int(np.round(decm * chunk))
  arr = np.clip(arr, smallest, largest)
  print('2->' + str(arr))
  return arr

# Function to generate training data batches
def simulate_training_data(batch_size, batch_num, seq_len, mean):
  input_batches = []
  label_batches = []
  for i in range(batch_num):
    input_batch = []
    label_batch = []
    for j in range(batch_size):
      input_seq = [TOKEN_CLS]
      input_seq.extend(generate_random(vocab_dict['a'], vocab_dict['h'], mean, seq_len-2).tolist())
      input_seq.append(TOKEN_SEP)
      label_seq = simulate_output(input_seq)

      print('input: ' + str(input_seq))
      print('label: ' + str(label_seq))

      # Convert input_seq to one hot matrix
      input_seq_oh = []
      for ii in input_seq:
        ii_oh = [0.0 for _ in range(D_MODEL)] # Input embedding dimension (or one-hot in this experiment) has to be equal to D_MODEL
        ii_oh[ii] = 1.0
        ii_oh[ii + 12] = 1.0
        ii_oh[ii + 24] = 1.0
        ii_oh[ii + 36] = 1.0
        input_seq_oh.append(ii_oh)
        oh_input_map[json.dumps(ii_oh)] = dict_vocab[ii]
      
      input_batch.append(input_seq_oh)
      label_batch.append(label_seq)
    input_batches.append(input_batch)
    label_batches.append(label_batch)
  return input_batches, label_batches

############################################################
# FUNCTIONS FOR CREATE AND TRAIN MODEL
def generate_position_embedding(input_len, d_model, max_len=5000):
  pos_enc = np.array([
    [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] 
    if pos != 0 else np.zeros(d_model) 
      for pos in range(max_len)
    ])
  pos_enc[0:, 0::2] = np.sin(pos_enc[0:, 0::2]) # dim 2i
  pos_enc[0:, 1::2] = np.cos(pos_enc[0:, 1::2]) # dim 2i+1
  pos_enc = pos_enc[:input_len,:]
  return pos_enc

# Transpose 2D tensor to [Batch, Head, Len, D_Model]
def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                          seq_length, width):
  output_tensor = tf.reshape(
      input_tensor, [batch_size, seq_length, num_attention_heads, width])
  output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
  return output_tensor

# Build attention head disagreement cost
def get_attention_heads_disagreement_cost(output_tensor):
    x = output_tensor # shape [batch, q_length, heads, channels]
    x = tf.nn.l2_normalize(x, dim=-1) # normalize the last dimension
    x1 = tf.expand_dims(x, 2)  # shape [batch, q_length, 1, heads, channels]
    x2 = tf.expand_dims(x, 3)  # shape [batch, q_length, heads, 1, channels]
    cos_diff = tf.reduce_sum(tf.multiply(x1, x2), axis=[-1]) # shape [batch, q_length, heads, heads], broadcasting
    # cos_diff_square = tf.reduce_mean(tf.square(cos_diff), axis=[-2,-1])
    cos_diff = tf.reduce_mean(cos_diff, axis=[-2,-1]) + 1.0  #shape [batch, q_length]
    return cos_diff

# Build simple model with single Multi-Head Attention layer
def build_model(batch, seq_len, vocab_size, d_model, head):
  input_tensor = tf.placeholder(shape=(batch, seq_len, d_model), dtype=tf.int32)
  mask_tensor = tf.placeholder(shape=(batch, seq_len), dtype=tf.float32)

  # We are not using embedding here
  input_ids = tf.cast(input_tensor, tf.float32)

  # Add positional encoding. We use static positional encoding here.
  if USE_POSITIONAL_ENCODING:
    pos_enc = generate_position_embedding(input_len=seq_len, d_model=d_model)
    pos_enc = tf.constant(pos_enc, dtype=tf.float32)
    input_ids = input_ids + pos_enc

  # Convert input to 2D tensor
  input_batch = tf.reshape(input_ids, (-1, d_model))

  # Transform input to Q, K and V tensor
  size_per_head = int(d_model / head)
  K = tf.layers.dense(input_batch, size_per_head * head, name='K')
  Q = tf.layers.dense(input_batch, size_per_head * head, name='Q')
  V = tf.layers.dense(input_batch, size_per_head * head, name='V')

  # [Batch, Head, Len, Size_per_Head]
  K = transpose_for_scores(K, batch, head, seq_len, size_per_head)
  Q = transpose_for_scores(Q, batch, head, seq_len, size_per_head)
  V = transpose_for_scores(V, batch, head, seq_len, size_per_head)

  # Scaled Dot-Product attention [Batch, Head, Len-Q, Len-K]
  attention_scores = tf.matmul(Q, K, transpose_b=True)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  # Generate attention mask to prevent attention to padding tokens
  to_mask = tf.reshape(mask_tensor, [batch, 1, seq_len])
  broadcast_ones = tf.ones(
      shape=[batch, seq_len, 1], dtype=tf.float32)
  # Attention mask [Batch, Len, Len]
  attention_mask = broadcast_ones * to_mask
  # `attention_mask` = [Batch, 1, Len, Len]
  attention_mask = tf.expand_dims(attention_mask, axis=[1])
  # Make adding -10000.0 to attention of padding tokens
  adder = (1.0 - attention_mask) * -10000.0
  attention_scores += adder

  # `attention_probs` = [Batch, Head, Len, Len]
  attention_probs = tf.nn.softmax(attention_scores)

  # `context_layer` = [Batch, Head, Len-Q, Size_per_Head]
  context_layer = tf.matmul(attention_probs, V)

  # `context_layer` = [Batch, Len-Q, Head, Size_per_Head]
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  # Also calculate cost of attention head output difference here.
  disagreement_cost = get_attention_heads_disagreement_cost(context_layer)

  # `output_tensor` = [Batch x Len-Q, Head x Size_per_Head = D_Model]
  output_tensor = tf.reshape(
      context_layer,
      [batch * seq_len, head * size_per_head])

  # Final linear projection. Note that this weight has permutation set divided by row instead of column as in K/Q/V
  output_tensor = tf.layers.dense(
      output_tensor,
      d_model,
      name='output')

  # `output_tensor` = [Batch, Len-Q, Head x Size_per_Head = D_Model]
  output_tensor = tf.reshape(
      output_tensor,
      [batch, seq_len, head * size_per_head])

  # Pooled output is the 1st dimension of each hidden state of all tokens
  pooled_output_tensor = tf.reduce_mean(output_tensor, axis=-1) # output_tensor[:, :, 0]

  # Add binary classification layers
  logprob_tensor = tf.nn.sigmoid(pooled_output_tensor, name ='sigmoid')

  return (input_tensor, mask_tensor, pooled_output_tensor, disagreement_cost, logprob_tensor, attention_probs)
     
# Build loss graph to evaluate the model
def build_loss_graph(output_tensor, batch, seq_len, d_model, additional_costs):
  label_tensor = tf.placeholder(shape=output_tensor.get_shape(), dtype=tf.float32)
  classification_losses = tf.losses.sigmoid_cross_entropy(label_tensor, output_tensor)
  total_loss = classification_losses # + tf.reduce_mean(additional_costs)
  return (label_tensor, total_loss, classification_losses)

# Build training graph to optimize the loss
def build_train_graph(output_tensor, batch, seq_len, d_model, additional_costs):
  label_tensor, loss, classification_loss = build_loss_graph(output_tensor, batch, seq_len, d_model, additional_costs)
  optimizer = tf.train.GradientDescentOptimizer(0.0001)
  train_op = optimizer.minimize(loss)
  return (label_tensor, train_op, loss, classification_loss)

# Get all model weights from current graph
def get_all_variables(sess):
  with tf.variable_scope('K', reuse=True):
    K_kernel = sess.run(tf.get_variable('kernel'))
    K_bias = sess.run(tf.get_variable('bias'))
  with tf.variable_scope('Q', reuse=True):
    Q_kernel = sess.run(tf.get_variable('kernel'))
    Q_bias = sess.run(tf.get_variable('bias'))
  with tf.variable_scope('V', reuse=True):
    V_kernel = sess.run(tf.get_variable('kernel'))
    V_bias = sess.run(tf.get_variable('bias'))
  with tf.variable_scope('output', reuse=True):
    output_kernel = sess.run(tf.get_variable('kernel'))
    output_bias = sess.run(tf.get_variable('bias'))
  return [K_kernel, K_bias, Q_kernel, Q_bias, V_kernel, V_bias, output_kernel, output_bias]

# Set all model weights to current graph
def set_all_variables(sess, var_list):
  [K_kernel, K_bias, Q_kernel, Q_bias, V_kernel, V_bias, output_kernel, output_bias] = var_list
  with tf.variable_scope('K', reuse=True):
    sess.run(tf.get_variable('kernel').assign(K_kernel))
    sess.run(tf.get_variable('bias').assign(K_bias))
  with tf.variable_scope('Q', reuse=True):
    sess.run(tf.get_variable('kernel').assign(Q_kernel))
    sess.run(tf.get_variable('bias').assign(Q_bias))
  with tf.variable_scope('V', reuse=True):
    sess.run(tf.get_variable('kernel').assign(V_kernel))
    sess.run(tf.get_variable('bias').assign(V_bias))
  with tf.variable_scope('output', reuse=True):
    sess.run(tf.get_variable('kernel').assign(output_kernel))
    sess.run(tf.get_variable('bias').assign(output_bias))

# Run an evaluation on a model initialized with the specified weights
# Output of the model will be all weights of the trained model
def test_a_model(input_seq, mask_seq, label_seq, var_list, d_model, head, print_output=False):
  # Clear all stuffs in default graph, so we can start fresh
  tf.reset_default_graph()

  batch_size = len(input_seq[0])
  seq_len = len(input_seq[0][0])

  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  (input_tensor, mask_tensor, output_tensor, disagreement_cost, logprob_tensor, attention_probs_tensor) = build_model(batch=batch_size, seq_len=seq_len, vocab_size=VOCAB_SIZE, d_model=d_model, head=head)
  (label_tensor, loss, classification_loss) = build_loss_graph(output_tensor=output_tensor, batch=batch_size, seq_len=seq_len, d_model=d_model, additional_costs=[disagreement_cost])
  sess.run(tf.global_variables_initializer())
  set_all_variables(sess, var_list)

  avg_loss = 0.0
  avg_disgreement_loss = 0.0
  avg_classification_loss = 0.0
  avg_accuracy = 0.0
  sampled_attention_probs = None
  sampled_input_vals = None
  sampled_logprob_vals = None
  for input_sample, mask_sample, label_sample in zip(input_seq, mask_seq, label_seq):
    [output_vals, loss_vals, disagreement_cost_vals, classification_loss_vals, logprob_vals, attention_probs] = sess.run([output_tensor, loss, disagreement_cost, classification_loss, logprob_tensor, attention_probs_tensor], feed_dict={input_tensor: input_sample, mask_tensor: mask_sample, label_tensor: label_sample})
    '''
    print('----------------------------------------------------------------------')
    for input_ids, output_v, label_v in zip(input_sample, logprob_vals, label_sample) :
      input_decoded = decode_input_ids(input_ids)
      print(' --> ' + str(input_decoded) + ' => ' + str(output_v) + '/' + str(label_v))
    print('----------------------------------------------------------------------')
    '''
    avg_loss = avg_loss + loss_vals
    avg_disgreement_loss = avg_disgreement_loss + disagreement_cost_vals
    avg_classification_loss = avg_classification_loss + classification_loss_vals
    labels = np.array(label_sample)
    predictions = (logprob_vals >= 0.5).astype(int)
    scores = (predictions == labels).astype(int)
    scores = np.average(scores)
    avg_accuracy = avg_accuracy + scores
    sampled_attention_probs = attention_probs
    sampled_input_vals = decode_input_oh_batch(input_sample)
    sampled_logprob_vals = logprob_vals
  avg_loss = avg_loss / len(input_seq)
  avg_disgreement_loss = avg_disgreement_loss / len(input_seq)
  avg_classification_loss = avg_classification_loss / len(input_seq)
  avg_accuracy = avg_accuracy / len(input_seq)

  if print_output:
    print('=== Input Values ===')
    print(input_seq)
    print('=== Label Values ===')
    print(label_seq)
    print('=== Output Values ===')
    print(output_vals)
    print('=== Loss Values ===')
    print(avg_loss)
    print('=== Classification Loss Values ===')
    print(avg_classification_loss)
    print('=== Disagreement Loss Values ===')
    print(avg_disgreement_loss)
    print('=== Accuracy ===')
    print(avg_accuracy)
  return avg_loss, avg_disgreement_loss, avg_classification_loss, avg_accuracy, sampled_input_vals, sampled_attention_probs, sampled_logprob_vals

# Run an experiment by initial new model and perform training for 10 steps.
# Output of the model will be all weights of the trained model
def train_a_model(input_seq, mask_seq, label_seq, vocab_size, d_model, head, init_weights, print_output=False):
  # Clear all stuffs in default graph, so we can start fresh
  tf.reset_default_graph()

  batch_size = len(input_seq[0])
  seq_len = len(input_seq[0][0])

  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  (input_tensor, mask_tensor, output_tensor, disagreement_cost, logprob_tensor, attention_probs_tensor) = build_model(batch=batch_size, seq_len=seq_len, vocab_size=vocab_size, d_model=d_model, head=head)
  (label_tensor, train_op, loss, classification_loss) = build_train_graph(output_tensor=output_tensor, batch=batch_size, seq_len=seq_len, d_model=d_model, additional_costs=[disagreement_cost])
  sess.run(tf.global_variables_initializer())

  if init_weights is not None:
    set_all_variables(sess, init_weights)

  for i in range(LOCAL_TRAIN_EPOCH):
    avg_loss = 0.0
    avg_disagreement_loss = 0.0
    avg_classification_loss = 0.0
    avg_accuracy = 0.0
    for input_sample, mask_sample, label_sample in zip(input_seq, mask_seq, label_seq):
      [output_vals, loss_vals, disagreement_cost_vals, classification_loss_vals, logprob_vals, attention_probs, _] = sess.run([output_tensor, loss, disagreement_cost, classification_loss, logprob_tensor, attention_probs_tensor, train_op], feed_dict={input_tensor: input_sample, mask_tensor: mask_sample, label_tensor: label_sample})
      avg_loss = avg_loss + loss_vals
      avg_disagreement_loss = avg_disagreement_loss + disagreement_cost_vals
      avg_classification_loss = avg_classification_loss + classification_loss_vals
      labels = np.array(label_sample)
      predictions = (logprob_vals >= 0.5).astype(int)
      scores = (predictions == labels).astype(int)
      scores = np.average(scores)
      avg_accuracy = avg_accuracy + scores
    avg_loss = avg_loss / len(input_seq)
    avg_disagreement_loss = avg_disagreement_loss / len(input_seq)
    avg_classification_loss = avg_classification_loss / len(input_seq)
    avg_accuracy = avg_accuracy / len(input_seq)
    if print_output:
      print('EPOCH: ' + str(i))

  if print_output:
    print('=== Input Values ===')
    print(input_seq)
    print('=== Label Values ===')
    print(label_seq)
    print('=== Output Values ===')
    print(output_vals)
    print('=== Loss Values ===')
    print(avg_loss)
    print('=== Classification Loss Values ===')
    print(avg_classification_loss)
    print('=== Disagreement Loss Values ===')
    print(avg_disagreement_loss)
    print('=== Accuracy ===')
    print(avg_accuracy)

  trained_weights = get_all_variables(sess)
  return [avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy, trained_weights]

############################################################
# FUNCTIONS FOR FEDERATED LEARNING AND WEIGHT MATCHINGS

# Split and transform variable list of the model into 2 groups.
# 1) Weights those need permutation split by column
# 2) Weights those do not need permutation split
def split_weights_for_perm(var_list):
  perm_list = [
    var_list[0], # K_kernel
    var_list[1], # K_bias
    var_list[2], # Q_kernel
    var_list[3], # Q_bias
    var_list[4], # V_kernel
    var_list[5], # V_bias
    var_list[6].transpose(), # output_kernel (permutated rowwise)
  ]
  non_perm_list = [
    var_list[7], # output_bias (no permutation needed)
  ]
  return [perm_list, non_perm_list]

# Join permutated list back to variable list
def join_weights_from_perm(perm_list, non_perm_list):
  var_list = [
    perm_list[0], # K_kernel
    perm_list[1], # K_bias
    perm_list[2], # Q_kernel
    perm_list[3], # Q_bias
    perm_list[4], # V_kernel
    perm_list[5], # V_bias
    perm_list[6].transpose(), # output_kernel (permutated rowwise)
    non_perm_list[0], # output_bias (no permutation needed)
  ]
  return var_list

# Transpose K,Q,V weight matrix to [ATTENTION_HEAD, D_MODEL, KEY_SIZE]
def transpose_for_matching(K_val, head, size_per_head):
  print('Weight (Original)')
  print(K_val)
  print('Weight shape (Original)')
  print(K_val.shape)

  # [D_MODEL, ATTENTION_HEAD, KEY_SIZE]
  K_val = np.reshape(K_val, [-1, head, size_per_head])
  print('Weight (Splited)')
  print(K_val)
  print('Weight shape (Splited)')
  print(K_val.shape)

  # [ATTENTION_HEAD, D_MODEL, KEY_SIZE]
  K_val = np.transpose(K_val, [1, 0, 2])
  print('Weight (Transposed)')
  print(K_val)
  print('Weight shape (Transposed)')
  print(K_val.shape)
  return K_val

# Transpose K,Q,V weight matrix back to [D_MODEL, KEY_SIZE * ATTENTION_HEAD]
def transpose_back_from_matching(K_val, head, size_per_head):
  print('Weight (Before)')
  print(K_val)
  print('Weight shape (Before)')
  print(K_val.shape)

  # [D_MODEL, ATTENTION_HEAD, KEY_SIZE]
  K_val = np.transpose(K_val, [1, 0, 2])
  print('Weight (transposed)')
  print(K_val)
  print('Weight shape (transposed)')
  print(K_val.shape)

  # [D_MODEL, ATTENTION_HEAD x KEY_SIZE]
  K_val = np.reshape(K_val, [-1, head * size_per_head])
  # Handle case of single vector, not matrix
  if K_val.shape[0] == 1:
    K_val = np.reshape(K_val, -1)
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
  return [np.array([input_list[i] for i in perm_mat]) for input_list in perm_set]

def distance_function_euclidian(list1, list2):
  acc_dist = 0.0
  for a, b in zip(list1, list2):
    acc_dist = acc_dist + np.sum(np.abs(a - b))
  print('Distance = ' + str(acc_dist))
  return acc_dist

def distance_function_cosine(list1, list2):
  acc_dist = 0.0
  for a, b in zip(list1, list2):
    cos_dist = np.inner(a.flatten(), b.flatten())
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    acc_dist = acc_dist + cos_dist / norm
  acc_dist = -acc_dist
  print('Distance = ' + str(acc_dist))
  return acc_dist

distance_function = distance_function_euclidian
if MATCH_USING_COSINE_SIMILARITY:
  distance_function = distance_function_cosine

def find_best_permutation_matrix(list1, list2):
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

def calculate_federated_weights(list1, list2):
  return [np.average(np.array([a, b]), axis=0) for a, b in zip(list1, list2)]

# Function to apply federated node matching algorithm on 2 local weights
def perform_weight_permutation_matching(node_weights, d_model, head):
  size_per_head = int(d_model / head)
  node_weights_splitted = [split_weights_for_perm(b) for b in node_weights] 
  node_weights_transposed = [[transpose_for_matching(w, head=head, size_per_head=size_per_head) for w in b[0]] for b in node_weights_splitted]
  min_perm_mat, min_distance = find_best_permutation_matrix(node_weights_transposed[0], node_weights_transposed[1])
  node_weights_transposed[0] = apply_permutation_matrix(node_weights_transposed[0], min_perm_mat)
  node_weights_transposed = [[transpose_back_from_matching(w, head=head, size_per_head=size_per_head) for w in b] for b in node_weights_transposed]
  for i in range(len(node_weights_transposed)):
    node_weights_splitted[i][0] = node_weights_transposed[i]
  node_weights_perm = [join_weights_from_perm(b[0], b[1]) for b in node_weights_splitted] 
  return node_weights_perm, min_perm_mat, min_distance

# Perform 1 round of federated training, each local node is inited with federated_weights.
# This function returns updated weights from each local node along with their training metrics.
def perform_1_federated_training_round(input_seqs, mask_seqs, label_seqs, vocab_size, d_model, head, node_count, federated_weights):
  # Run local training and collect Loss and Trained Weights
  node_losses = []
  node_disgreement_losses = []
  node_classification_losses = []
  node_accuracy = []
  node_weights = []
  for i in range(NODE_COUNT):
    loss, disagreement_loss, classification_loss, accuracy, trained_weights = train_a_model(input_seqs[i], mask_seqs[i], label_seqs[i], vocab_size, d_model, head, federated_weights)
    node_losses.append(loss)
    node_disgreement_losses.append(disagreement_loss)
    node_classification_losses.append(classification_loss)
    node_accuracy.append(accuracy)
    node_weights.append(trained_weights)
  return node_losses, node_disgreement_losses, node_classification_losses, node_accuracy, node_weights

# Function to save weight log to file, for displaying later
def save_weight_logs(node_weights, epoch, algor):
  if not os.path.exists('weight_logs'):
    os.makedirs('weight_logs')
  file_path = os.path.join('weight_logs', '19_benchmark_' + algor + '_trial_' + str(current_trial_round) + '_' + str(epoch) + '.pkl')
  with open(file_path, 'wb') as fout:
    pickle.dump(node_weights, fout)

# Function to save attention score of sampled test data, for displaying later
def save_attention_score_logs(attention_scores, epoch, algor):
  if not os.path.exists('attention_logs'):
    os.makedirs('attention_logs')
  file_path = os.path.join('attention_logs', '19_benchmark_' + algor + '_trial_' + str(current_trial_round) + '_' + str(epoch) + '.pkl')
  with open(file_path, 'wb') as fout:
    pickle.dump(attention_scores, fout)


# ================================
# Starting of the benchmark
# ================================
for trial in range(TRIAL_NUM):
  current_trial_round = trial

  # Simulate input and label.
  # Both federated node see training data from different distribution of X
  input_seqs = []
  label_seqs = []
  mask_seqs = []
  mean_x_vals = [vocab_dict['b'], vocab_dict['g']] # Mean of X value for each local training data
  for i in range(NODE_COUNT):  
    input_seq, label_seq = simulate_training_data(batch_size=BATCH_SIZE, 
      batch_num=BATCH_NUM, 
      seq_len=SEQ_LEN,
      mean=mean_x_vals[i]
      )
    print('-------------------------------------------')
    print('Local training data for Federated Node: ' + str(i))
    print('-------------------------------------------')
    print(np.array(input_seq).shape)
    print(np.array(label_seq).shape)

    # Count number of 1.0 and 0.0 for the dataset
    count_positive = 0
    count_all = 0
    for label_batch in label_seq:
      for label in label_batch:
        label_pooled = label[0]
        count_all = count_all + 1
        if label_pooled == 1.0:
          count_positive = count_positive + 1
    # print('input_seq: X[0] has average values = ' + str(np.average(np.array(input_seq)[0,:,:])))
    print('label_seq: Y[0] has positive case of ' + str(count_positive) + '/' + str(count_all)  + ' = ' + str(count_positive * 100 / count_all) + '%')

    input_seqs.append(input_seq)
    label_seqs.append(label_seq)
    mask_seqs.append(np.ones((BATCH_NUM, BATCH_SIZE, SEQ_LEN), dtype=np.float))

  print(np.array(input_seqs).shape)
  print(np.array(label_seqs).shape)
  print(np.array(mask_seqs).shape)

  # Test data is randomly picked from training data of both node equally
  test_input_seqs = []
  test_label_seqs = []
  test_mask_seqs = []
  for i in range(BATCH_NUM):
    '''
    choice = int(random.random() * NODE_COUNT)
    test_input_seqs.append(input_seqs[choice][i])
    test_label_seqs.append(label_seqs[choice][i])
    test_mask_seqs.append(mask_seqs[choice][i])
    '''
    test_input_seqs.append(input_seqs[0][i])
    test_label_seqs.append(label_seqs[0][i])
    test_mask_seqs.append(mask_seqs[0][i])
    test_input_seqs.append(input_seqs[1][i])
    test_label_seqs.append(label_seqs[1][i])
    test_mask_seqs.append(mask_seqs[1][i])
  print('-------------------------------------------')
  print('Global test data')
  print('-------------------------------------------')
  print('input_seq: X[0] has average values = ' + str(np.average(np.array(test_input_seqs)[:,0,:,0])))

  fedAVG_weights = None
  fedAVG_train_loss_history = []
  fedAVG_train_disagreement_loss_history = []
  fedAVG_train_classification_loss_history = []
  fedAVG_train_accuracy_history = []
  fedAVG_test_loss_history = []
  fedAVG_test_disagreement_loss_history = []
  fedAVG_test_classification_loss_history = []
  fedAVG_test_accuracy_history = []

  matched_fedAVG_weights = None
  matched_fedAVG_train_loss_history = []
  matched_fedAVG_train_disagreement_loss_history = []
  matched_fedAVG_train_classification_loss_history = []
  matched_fedAVG_train_accuracy_history = []
  matched_fedAVG_test_loss_history = []
  matched_fedAVG_test_disagreement_loss_history = []
  matched_fedAVG_test_classification_loss_history = []
  matched_fedAVG_test_accuracy_history = []
  min_perm_mats = []
  min_distances = []

  # Run 1 initial local training steps, so that each algorithm starts from the same local models aggregation
  initial_node_losses, inital_node_disagreement_losses, inital_node_classification_losses, initial_node_accuracy, initial_node_weights = perform_1_federated_training_round(
    input_seqs, 
    mask_seqs,
    label_seqs, 
    VOCAB_SIZE,
    D_MODEL, 
    ATTENTION_HEAD, 
    NODE_COUNT, 
    None
  )
  save_weight_logs(initial_node_weights, 0, 'fedma')
  save_weight_logs(initial_node_weights, 0, 'mfedma')
  fedAVG_train_loss_history.append(initial_node_losses)
  matched_fedAVG_train_loss_history.append(initial_node_losses)
  fedAVG_train_disagreement_loss_history.append(inital_node_disagreement_losses)
  matched_fedAVG_train_disagreement_loss_history.append(inital_node_disagreement_losses)
  fedAVG_train_classification_loss_history.append(inital_node_classification_losses)
  matched_fedAVG_train_classification_loss_history.append(inital_node_classification_losses)
  fedAVG_train_accuracy_history.append(initial_node_accuracy)
  matched_fedAVG_train_accuracy_history.append(initial_node_accuracy)

  # Run experiments on FedAVG aggregation method
  fedAVG_weights = calculate_federated_weights(initial_node_weights[0], initial_node_weights[1])
  avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy, sampled_input_vals, sampled_attention_probs, sampled_logprob_vals = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, fedAVG_weights, D_MODEL, ATTENTION_HEAD)
  fedAVG_test_loss_history.append(avg_loss)
  fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
  fedAVG_test_classification_loss_history.append(avg_classification_loss)
  fedAVG_test_accuracy_history.append(avg_accuracy)
  save_attention_score_logs([sampled_input_vals, sampled_attention_probs, sampled_logprob_vals], 0, 'fedma')

  for i in range(COMMUNICATION_ROUNDS):
    node_losses, node_disagreement_losses, node_classification_losses, node_accuracy, node_weights = perform_1_federated_training_round(
      input_seqs, 
      mask_seqs,
      label_seqs, 
      VOCAB_SIZE,
      D_MODEL, 
      ATTENTION_HEAD, 
      NODE_COUNT, 
      fedAVG_weights
    )
    save_weight_logs(node_weights, i + 1, 'fedma')
    fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
    fedAVG_train_loss_history.append(node_losses)
    fedAVG_train_disagreement_loss_history.append(node_disagreement_losses)
    fedAVG_train_classification_loss_history.append(node_classification_losses)
    fedAVG_train_accuracy_history.append(node_accuracy)
    avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy, sampled_input_vals, sampled_attention_probs, sampled_logprob_vals = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, fedAVG_weights, D_MODEL, ATTENTION_HEAD)
    fedAVG_test_loss_history.append(avg_loss)
    fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
    fedAVG_test_classification_loss_history.append(avg_classification_loss)
    fedAVG_test_accuracy_history.append(avg_accuracy)
    save_attention_score_logs([sampled_input_vals, sampled_attention_probs, sampled_logprob_vals], i + 1, 'fedma')

  # Run experiments on Matched FedAVG aggregation method
  node_weights, min_perm_mat, min_distance = perform_weight_permutation_matching(initial_node_weights, D_MODEL, ATTENTION_HEAD)
  matched_fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
  avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy, sampled_input_vals, sampled_attention_probs, sampled_logprob_vals = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, matched_fedAVG_weights, D_MODEL, ATTENTION_HEAD)
  matched_fedAVG_test_loss_history.append(avg_loss)
  matched_fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
  matched_fedAVG_test_classification_loss_history.append(avg_classification_loss)
  matched_fedAVG_test_accuracy_history.append(avg_accuracy)
  min_perm_mats.append(min_perm_mat)
  min_distances.append(min_distance)
  save_attention_score_logs([sampled_input_vals, sampled_attention_probs, sampled_logprob_vals], 0, 'mfedma')

  for i in range(COMMUNICATION_ROUNDS):
    node_losses, node_disagreement_losses, node_classification_losses, node_accuracy, node_weights = perform_1_federated_training_round(
      input_seqs, 
      mask_seqs,
      label_seqs, 
      VOCAB_SIZE,
      D_MODEL, 
      ATTENTION_HEAD, 
      NODE_COUNT, 
      matched_fedAVG_weights
    )
    save_weight_logs(node_weights, i + 1, 'mfedma')
    # Calculate the best permutation matrix to match the weights from 2 nodes
    node_weights, min_perm_mat, min_distance = perform_weight_permutation_matching(node_weights, D_MODEL, ATTENTION_HEAD)
    matched_fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
    matched_fedAVG_train_loss_history.append(node_losses)
    matched_fedAVG_train_disagreement_loss_history.append(node_disagreement_losses)
    matched_fedAVG_train_classification_loss_history.append(node_classification_losses)
    matched_fedAVG_train_accuracy_history.append(node_accuracy)
    avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy, sampled_input_vals, sampled_attention_probs, sampled_logprob_vals = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, matched_fedAVG_weights, D_MODEL, ATTENTION_HEAD)
    matched_fedAVG_test_loss_history.append(avg_loss)
    matched_fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
    matched_fedAVG_test_classification_loss_history.append(avg_classification_loss)
    matched_fedAVG_test_accuracy_history.append(avg_accuracy)
    min_perm_mats.append(min_perm_mat)
    min_distances.append(min_distance)
    save_attention_score_logs([sampled_input_vals, sampled_attention_probs, sampled_logprob_vals], i + 1, 'mfedma')

  # Print experiemental results
  for i in range(COMMUNICATION_ROUNDS + 1):
    print('Comm Round: ' + str(i))

    min_perm_mat = min_perm_mats[i]
    min_distance = min_distances[i]
    print(' Min permutation matrix = ' + str(min_perm_mat) + '\t distance = ' + str(min_distance) + '\n')

    train_loss = fedAVG_train_loss_history[i]
    test_loss = fedAVG_test_loss_history[i]
    train_acc = fedAVG_train_accuracy_history[i]
    test_acc = fedAVG_test_accuracy_history[i]
    print(' FedAVG \t Train Loss: ' + str(train_loss)+ '\t Test Loss: ' + str(test_loss) + '\t Train Acc: ' + str(train_acc) + '\t Test Acc: ' + str(test_acc))

    train_loss = matched_fedAVG_train_loss_history[i]
    test_loss = matched_fedAVG_test_loss_history[i]
    train_acc = matched_fedAVG_train_accuracy_history[i]
    test_acc = matched_fedAVG_test_accuracy_history[i]
    print(' MFedAVG \t rain Loss: ' + str(train_loss)+ '\t Test Loss: ' + str(test_loss) + '\t Train Acc: ' + str(train_acc) + '\t Test Acc: ' + str(test_acc))

  # Save output to log file
  if not os.path.exists('output_logs'):
    os.makedirs('output_logs')
  with open(os.path.join('output_logs', '19_output'  + '_trial_' + str(current_trial_round)+ '.csv'), 'w', encoding='utf-8') as fout:
    fout.write('Federated Round,' +
      'FedAVG Local Loss 1,FedAVG Local Loss 2,Matched FedAVG Local Loss 1,Matched FedAVG Local Loss 2,' +
      'FedAVG Local Disagreement Loss 1,FedAVG Local Disagreement Loss 2,Matched FedAVG Local Disagreement Loss 1,Matched FedAVG Local Disagreement Loss 2,' +
      'FedAVG Local Classification Loss 1,FedAVG Local Classification Loss 2,Matched FedAVG Local Classification Loss 1,Matched FedAVG Local Classification Loss 2,' +
      'FedAVG Local Accuracy 1,FedAVG Local Accuracy 2,Matched FedAVG Local Accuracy 1,Matched FedAVG Local Accuracy 2,' +
      'FedAVG Global Loss,Matched FedAVG Global Loss,' +
      'FedAVG Global Disagreement Loss,Matched FedAVG Global Disagreement Loss,' +
      'FedAVG Global Classification Loss,Matched FedAVG Global Classification Loss,' +
      'FedAVG Global Accuracy,Matched FedAVG Global Accuracy' +
      '\n')
    for i in range(COMMUNICATION_ROUNDS + 1):
      fedAVG_train_loss = fedAVG_train_loss_history[i]
      fedAVG_test_loss = fedAVG_test_loss_history[i]
      matched_fedAVG_train_loss = matched_fedAVG_train_loss_history[i]
      matched_fedAVG_test_loss = matched_fedAVG_test_loss_history[i]

      fedAVG_train_disagreement_loss = [np.average(a) for a in fedAVG_train_disagreement_loss_history[i]]
      fedAVG_test_disagreement_loss = [np.average(a) for a in fedAVG_test_disagreement_loss_history[i]][0]
      matched_fedAVG_train_disagreement_loss = [np.average(a) for a in matched_fedAVG_train_disagreement_loss_history[i]]
      matched_fedAVG_test_disagreement_loss = [np.average(a) for a in matched_fedAVG_test_disagreement_loss_history[i]][0]

      fedAVG_train_classification_loss = fedAVG_train_classification_loss_history[i]
      fedAVG_test_classification_loss = fedAVG_test_classification_loss_history[i]
      matched_fedAVG_train_classification_loss = matched_fedAVG_train_classification_loss_history[i]
      matched_fedAVG_test_classification_loss = matched_fedAVG_test_classification_loss_history[i]

      fedAVG_train_accuracy = fedAVG_train_accuracy_history[i]
      fedAVG_test_accuracy = fedAVG_test_accuracy_history[i]
      matched_fedAVG_train_accuracy = matched_fedAVG_train_accuracy_history[i]
      matched_fedAVG_test_accuracy = matched_fedAVG_test_accuracy_history[i]

      fout.write(
        str(i) + ',' + 
        str(fedAVG_train_loss[0]) + ',' + 
        str(fedAVG_train_loss[1]) + ',' + 
        str(matched_fedAVG_train_loss[0]) + ',' + 
        str(matched_fedAVG_train_loss[1]) + ',' + 
        str(fedAVG_train_disagreement_loss[0]) + ',' + 
        str(fedAVG_train_disagreement_loss[1]) + ',' + 
        str(matched_fedAVG_train_disagreement_loss[0]) + ',' + 
        str(matched_fedAVG_train_disagreement_loss[1]) + ',' + 
        str(fedAVG_train_classification_loss[0]) + ',' + 
        str(fedAVG_train_classification_loss[1]) + ',' + 
        str(matched_fedAVG_train_classification_loss[0]) + ',' + 
        str(matched_fedAVG_train_classification_loss[1]) + ',' + 
        str(fedAVG_train_accuracy[0]) + ',' + 
        str(fedAVG_train_accuracy[1]) + ',' + 
        str(matched_fedAVG_train_accuracy[0]) + ',' + 
        str(matched_fedAVG_train_accuracy[1]) + ',' + 
        str(fedAVG_test_loss) + ',' + 
        str(matched_fedAVG_test_loss) + ',' +
        str(fedAVG_test_disagreement_loss) + ',' + 
        str(matched_fedAVG_test_disagreement_loss) + ',' +
        str(fedAVG_test_classification_loss) + ',' + 
        str(matched_fedAVG_test_classification_loss) + ',' +
        str(fedAVG_test_accuracy) + ',' + 
        str(matched_fedAVG_test_accuracy) +
        '\n')

  print('Finished running trial: ' + str(current_trial_round))

print('Finished benchmarking.')

