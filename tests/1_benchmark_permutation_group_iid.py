import os
import sys
import numpy as np
import tensorflow as tf
import math
import random

# This benchmark is for compare between using 2 federated nodes training with single Dot Product Attention layer.
# It shows how much the different between using weight permutation matching and simple FedAVG algorithm.
# For this benchmark, we will implement 2-Heads Self Attenton layer and let 2 model trained from simulated data.
# Then we perform weight aggregation and benchmark how well it perform in test data.

# For this experiment, we perform training with IID training data and test data.
# Because the data is IID, training data for both federated nodes and test data are generated from single 
# simulation function.

# Simulated function
# The function take 6 input tokens, each token has 2 dimension
#  - The first dimension stores simple float number from -1.0 - 1.0
#  - The second dimension stores simulated position embedding. The value will be 0.1 for position 0 and 1.0 for position 9

LOCAL_TRAIN_EPOCH = 10
ATTENTION_HEAD = 2
BATCH_SIZE = 1
BATCH_NUM = 10
D_MODEL = 2
SEQ_LEN = 6

# Number of federated nodes
NODE_COUNT = 2

####################################################################
# FUNCTION FOR SETUP RANDOMSEED SO THAT EXPERIMENTS ARE REPRODUCIBLE
RANDOM_SEED = 1234
def setup_random_seed(seed_value):
  # Set `PYTHONHASHSEED` environment variable at a fixed value
  os.environ['PYTHONHASHSEED'] = str(seed_value)
  # Set `python` built-in pseudo-random generator at a fixed value
  random.seed(seed_value)
  # Set `numpy` pseudo-random generator at a fixed value
  np.random.seed(seed_value)
  # Set `tensorflow` pseudo-random generator at a fixed value
  tf.set_random_seed(random.randint(0, 65535))

setup_random_seed(RANDOM_SEED)

####################################################################
# DETECT GPU WITH LOWEST LOADED AND USE THE GPU FOR ENTIRE PROGRAM
from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
NUM_GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
USED_DEVICE = None
if NUM_GPUS == 0:
  USED_DEVICE = '/device:CPU:0'
else:
  import subprocess as sp
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_infos = _output_to_list(sp.check_output(COMMAND.split()))
  memory_free_info = memory_free_infos[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  chosen_gpu = np.argmax(np.array(memory_free_values))
  print('[INFO] Use GPU: ' + str(chosen_gpu))
  USED_DEVICE = '/device:GPU:' + str(chosen_gpu)

with tf.device(USED_DEVICE):

  ####################################################################
  # FUNCTION FOR CREATE TENSORFLOW SESSION, USING GPU MEMORY AS NEEDED
  def setup_tensorflow_session():
    # Detect if we have GPU. CPU-Only enviroment should not use allow_growth
    if NUM_GPUS == 0:
      return tf.Session()
    
    # For GPU, we want to allocate GPU memory at minimal... 
    # so we can perform expeiments in parallel using many processes.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    return sess

  ############################################################
  # FUNCTIONS FOR CREATE AND TRAIN MODEL
  def add_position_embedding(input_seq, len):
    output_seq = np.zeros((len, 2))
    output_seq[0:len, 0] = input_seq
    output_seq[0:len, 1] = [float(i) / len for i in range(len)]
    return output_seq

  def simulate_output(input_seq):
    output_seq = np.zeros((input_seq.shape[0], 2))
    r = input_seq[:, 0] # + input_seq[:, 1]
    r = r - 0.5
    r = r * r
    output_seq[:, 0] = r
    output_seq[:, 1] = input_seq[:, 1]
    return output_seq

  # Transpose 2D tensor to [Batch, Head, Len, D_Model]
  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                            seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])
    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  # Build simple model with single Multi-Head Attention layer
  def build_model(batch, seq_len, d_model, head):
    input_tensor = tf.placeholder(shape=(batch, seq_len, d_model), dtype=tf.float32)

    # Convert input to 2D tensor
    input_batch = tf.reshape(input_tensor, (-1, d_model))

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
    attention_probs = tf.nn.softmax(attention_scores)

    # `context_layer` = [Batch, Head, Len-Q, Size_per_Head]
    context_layer = tf.matmul(attention_probs, V)

    # `context_layer` = [Batch, Len-Q, Head, Size_per_Head]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

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

    return (input_tensor, output_tensor)

  # Build loss graph to evaluate the model
  def build_loss_graph(output_tensor, batch, seq_len, d_model):
    label_tensor = tf.placeholder(shape=(batch, seq_len, d_model), dtype=tf.float32)
    loss = tf.losses.mean_squared_error(output_tensor, label_tensor)
    return (label_tensor, loss)

  # Build training graph to optimize the loss
  def build_train_graph(output_tensor, batch, seq_len, d_model):
    label_tensor, loss = build_loss_graph(output_tensor, batch, seq_len, d_model)
    optimizer = tf.train.GradientDescentOptimizer(0.0001)
    train_op = optimizer.minimize(loss)
    return (label_tensor, train_op, loss)

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
  def test_a_model(input_seq, label_seq, var_list, d_model, head, print_output=False):
    # Clear all stuffs in default graph, so we can start fresh
    tf.reset_default_graph()

    with tf.device(USED_DEVICE):
      # We want each session to have different random seed, but we need each run to have the same random sequence
      tf.set_random_seed(random.randint(0, 65535))

      batch_size = len(input_seq[0])
      seq_len = len(input_seq[0][0])

      sess = setup_tensorflow_session()
      (input_tensor, output_tensor) = build_model(batch=batch_size, seq_len=seq_len, d_model=d_model, head=head)
      (label_tensor, loss) = build_loss_graph(output_tensor=output_tensor, batch=batch_size, seq_len=seq_len, d_model=d_model)
      sess.run(tf.global_variables_initializer())
      set_all_variables(sess, var_list)

      avg_loss = 0.0
      for input_sample, label_sample in zip(input_seq, label_seq):
        [output_vals, loss_vals] = sess.run([output_tensor, loss], feed_dict={input_tensor: input_sample, label_tensor: label_sample})
        avg_loss = avg_loss + loss_vals
      avg_loss = avg_loss / len(input_seq)

      if print_output:
        print('=== Input Values ===')
        print(input_seq)
        print('=== Label Values ===')
        print(label_seq)
        print('=== Output Values ===')
        print(output_vals)
        print('=== Loss Values ===')
        print(avg_loss)
      return avg_loss

  # Run an experiment by initial new model and perform training for 10 steps.
  # Output of the model will be all weights of the trained model
  def train_a_model(input_seq, label_seq, d_model, head, print_output=False):
    # Clear all stuffs in default graph, so we can start fresh
    tf.reset_default_graph()

    with tf.device(USED_DEVICE):
      # We want each session to have different random seed, but we need each run to have the same random sequence
      tf.set_random_seed(random.randint(0, 65535))

      batch_size = len(input_seq[0])
      seq_len = len(input_seq[0][0])

      sess = setup_tensorflow_session()
      (input_tensor, output_tensor) = build_model(batch=batch_size, seq_len=seq_len, d_model=d_model, head=head)
      (label_tensor, train_op, loss) = build_train_graph(output_tensor=output_tensor, batch=batch_size, seq_len=seq_len, d_model=d_model)
      sess.run(tf.global_variables_initializer())

      trained_weights = get_all_variables(sess)

      for i in range(LOCAL_TRAIN_EPOCH):
        avg_loss = 0.0
        for input_sample, label_sample in zip(input_seq, label_seq):
          [output_vals, loss_vals, _] = sess.run([output_tensor, loss, train_op], feed_dict={input_tensor: input_sample, label_tensor: label_sample})
          avg_loss = avg_loss + loss_vals
        avg_loss = avg_loss / len(input_seq)
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

      trained_weights = get_all_variables(sess)
      return [avg_loss, trained_weights]

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

  def distance_function(list1, list2):
    acc_dist = 0.0
    for a, b in zip(list1, list2):
      acc_dist = acc_dist + np.sum(np.abs(a - b))
    print('Distance = ' + str(acc_dist))
    return acc_dist

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

  # Function to generate training data batches
  def simulate_training_data(batch_size, batch_num, seq_len):
    input_batches = []
    label_batches = []
    for i in range(batch_num):
      input_batch = []
      label_batch = []
      for j in range(batch_size):
        input_seq = [random.uniform(-1.0, 1.0) for _ in range(seq_len)]
        input_seq = add_position_embedding(input_seq, len(input_seq))
        label_seq = simulate_output(input_seq)
        input_batch.append(input_seq)
        label_batch.append(label_seq)
      input_batches.append(input_batch)
      label_batches.append(label_batch)
    return input_batches, label_batches

  # Simulate input and label
  '''
  input_seq = [0.0,0.2,0.4,0.6,0.8,1.0]
  input_seq = add_position_embedding(input_seq, len(input_seq))
  label_seq = simulate_output(input_seq)
  '''
  input_seq, label_seq = simulate_training_data(batch_size=BATCH_SIZE, batch_num=BATCH_NUM, seq_len=SEQ_LEN)
  print('input_seq')
  print(input_seq)
  print('label_seq')
  print(label_seq)

  # Run local training and collect Loss and Trained Weights
  node_losses = []
  node_weights = []
  for i in range(NODE_COUNT):
    loss, trained_weights = train_a_model(input_seq, label_seq, D_MODEL, ATTENTION_HEAD)
    node_losses.append(loss)
    node_weights.append(trained_weights)

  # Perform FedAVG and evaluate the aggregated model
  fedavg_weights = calculate_federated_weights(node_weights[0], node_weights[1])
  fedavg_loss = test_a_model(input_seq, label_seq, fedavg_weights, D_MODEL, ATTENTION_HEAD)

  # Calculate the best permutation matrix to match the weights from 2 nodes
  size_per_head = int(D_MODEL / ATTENTION_HEAD)
  node_weights_splitted = [split_weights_for_perm(b) for b in node_weights] 
  node_weights_transposed = [[transpose_for_matching(w, head=ATTENTION_HEAD, size_per_head=size_per_head) for w in b[0]] for b in node_weights_splitted]
  min_perm_mat, min_distance = find_best_permutation_matrix(node_weights_transposed[0], node_weights_transposed[1])
  node_weights_transposed[0] = apply_permutation_matrix(node_weights_transposed[0], min_perm_mat)
  node_weights_transposed = [[transpose_back_from_matching(w, head=ATTENTION_HEAD, size_per_head=size_per_head) for w in b] for b in node_weights_transposed]
  for i in range(len(node_weights_transposed)):
    node_weights_splitted[i][0] = node_weights_transposed[i]
  node_weights_perm = [join_weights_from_perm(b[0], b[1]) for b in node_weights_splitted] 
  matched_fedavg_weights = calculate_federated_weights(node_weights_perm[0], node_weights_perm[1])
  matched_fedavg_loss = test_a_model(input_seq, label_seq, matched_fedavg_weights, D_MODEL, ATTENTION_HEAD)

  # Print result log
  for i in range(NODE_COUNT):
    print('=======================================================')
    print('Local Model #' + str(i))
    print(' Weights = ' + str(node_weights[i]))
    print(' Loss = ' + str(node_losses[i]))

  print('=======================================================')
  print('FedAVG Model')
  print(' Weights = ' + str(fedavg_weights))
  print(' Loss = ' + str(fedavg_loss))

  print('=======================================================')
  print('Matched FedAVG Model')
  print(' Weights = ' + str(matched_fedavg_weights))
  print(' Loss = ' + str(matched_fedavg_loss))
  print(' Permutation Matrix = ' + str(min_perm_mat))
