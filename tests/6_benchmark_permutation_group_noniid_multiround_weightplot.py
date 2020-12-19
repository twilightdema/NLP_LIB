import os
import sys
import numpy as np
import tensorflow as tf
import math
import random
import pickle

# This benchmark is for compare between using 2 federated nodes training with single Dot Product Attention layer.
# It shows how much the different between using weight permutation matching and simple FedAVG algorithm.
# For this benchmark, we will implement 2-Heads Self Attenton layer and let 2 model trained from simulated data.
# Then we perform weight aggregation and benchmark how well it perform in test data.

# For this experiment, we perform training with Non-IID training data and test data.
# Non-IID data is simulated by using same simulation function mapping from X->Y, 
# but one federated node will see the data from different distribution of X than another node.
# The test data is basically randomly picked from both distribution of X equally.

# In additional, we perform federated averaging and local training for many rounds so that we
# see how different they are in multi-communication rounds.

# Simulated function
# The function take 6 input tokens, each token has 2 dimension
#  - The first dimension stores simple float number from -1.0 - 1.0
#  - The second dimension stores simulated position embedding. The value will be 0.1 for position 0 and 1.0 for position 9

COMMUNICATION_ROUNDS = 8
LOCAL_TRAIN_EPOCH = 10
ATTENTION_HEAD = 4
BATCH_SIZE = 1
BATCH_NUM = 10
D_MODEL = 16
SEQ_LEN = 6

# Number of federated nodes
NODE_COUNT = 2

####################################################################
# FUNCTION FOR SETUP RANDOMSEED SO THAT EXPERIMENTS ARE REPRODUCIBLE
RANDOM_SEED = 2345
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
  def add_position_embedding(input_seq, input_len, d_model, max_len=5000):
    pos_enc = np.array([
      [pos / np.power(10000, 2 * (j // 2) / d_model) for j in range(d_model)] 
      if pos != 0 else np.zeros(d_model) 
        for pos in range(max_len)
      ])
    pos_enc[0:, 0::2] = np.sin(pos_enc[0:, 0::2]) # dim 2i
    pos_enc[0:, 1::2] = np.cos(pos_enc[0:, 1::2]) # dim 2i+1
    pos_enc = pos_enc[:input_len,:]

    output_seq = np.array(input_seq)
    output_seq = output_seq + pos_enc
    return output_seq

  def simulate_output(input_seq):
    output_seq = np.zeros(input_seq.shape)
    r = input_seq[:, 0] # + input_seq[:, 1]
    r = r - 0.5
    r = r * r
    output_seq[:, 0] = r # Set every element equals to r
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
  def train_a_model(input_seq, label_seq, d_model, head, init_weights, print_output=False):
    # Clear all stuffs in default graph, so we can start fresh
    tf.reset_default_graph()
    # We want each session to have different random seed, but we need each run to have the same random sequence
    tf.set_random_seed(random.randint(0, 65535))

    batch_size = len(input_seq[0])
    seq_len = len(input_seq[0][0])

    sess = setup_tensorflow_session()
    (input_tensor, output_tensor) = build_model(batch=batch_size, seq_len=seq_len, d_model=d_model, head=head)
    (label_tensor, train_op, loss) = build_train_graph(output_tensor=output_tensor, batch=batch_size, seq_len=seq_len, d_model=d_model)
    sess.run(tf.global_variables_initializer())

    if init_weights is not None:
      set_all_variables(sess, init_weights)

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

  # Function to generate training data batches, taking mean_x_val to bias distribution of X
  def simulate_training_data(batch_size, batch_num, seq_len, d_model, mean_x_val):
    input_batches = []
    label_batches = []
    for i in range(batch_num):
      input_batch = []
      label_batch = []
      for j in range(batch_size):
        input_seq = [
          [random.uniform(-1.0 + mean_x_val, 1.0 + mean_x_val) for _ in range(d_model)]
          for _ in range(seq_len)
        ]
        input_seq = add_position_embedding(input_seq, len(input_seq), d_model)
        label_seq = simulate_output(input_seq)
        input_batch.append(input_seq)
        label_batch.append(label_seq)
      input_batches.append(input_batch)
      label_batches.append(label_batch)
    return input_batches, label_batches

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
    return node_weights_perm

  # Perform 1 round of federated training, each local node is inited with federated_weights.
  # This function returns updated weights from each local node along with their training metrics.
  def perform_1_federated_training_round(input_seqs, label_seqs, d_model, head, node_count, federated_weights):
    # Run local training and collect Loss and Trained Weights
    node_losses = []
    node_weights = []
    for i in range(NODE_COUNT):
      loss, trained_weights = train_a_model(input_seqs[i], label_seqs[i], d_model, head, federated_weights)
      node_losses.append(loss)
      node_weights.append(trained_weights)
    return node_losses, node_weights

  # Function to save weight log to file, for displaying later
  def save_weight_logs(node_weights, epoch, algor):
    if not os.path.exists('weight_logs'):
      os.makedirs('weight_logs')
    file_path = os.path.join('weight_logs', '6_benchmark_' + algor + '_' + str(epoch) + '.pkl')
    with open(file_path, 'wb') as fout:
      pickle.dump(node_weights, fout)

  # Simulate input and label.
  # Both federated node see training data from different distribution of X
  input_seqs = []
  label_seqs = []
  mean_x_vals = [-0.75, 0.75] # Mean of X value for each local training data
  for i in range(NODE_COUNT):  
    input_seq, label_seq = simulate_training_data(batch_size=BATCH_SIZE, 
      batch_num=BATCH_NUM, 
      seq_len=SEQ_LEN,
      d_model=D_MODEL,
      mean_x_val=mean_x_vals[i]
      )
    print('-------------------------------------------')
    print('Local training data for Federated Node: ' + str(i))
    print('-------------------------------------------')
    print('input_seq: X[0] has average values = ' + str(np.average(np.array(input_seq)[:,0,:,0])))
    input_seqs.append(input_seq)
    label_seqs.append(label_seq)

  print(np.array(input_seqs).shape)
  print(np.array(label_seqs).shape)

  # Test data is randomly picked from training data of both node equally
  test_input_seqs = []
  test_label_seqs = []
  for i in range(BATCH_NUM):
    choice = int(random.random() * NODE_COUNT)
    test_input_seqs.append(input_seqs[choice][i])
    test_label_seqs.append(label_seqs[choice][i])
  print('-------------------------------------------')
  print('Global test data')
  print('-------------------------------------------')
  print('input_seq: X[0] has average values = ' + str(np.average(np.array(test_input_seqs)[:,0,:,0])))

  fedAVG_weights = None
  fedAVG_train_loss_history = []
  fedAVG_test_loss_history = []
  matched_fedAVG_weights = None
  matched_fedAVG_train_loss_history = []
  matched_fedAVG_test_loss_history = []

  # Run 1 initial local training steps, so that each algorithm starts from the same local models aggregation
  initial_node_losses, initial_node_weights = perform_1_federated_training_round(
    input_seqs, 
    label_seqs, 
    D_MODEL, 
    ATTENTION_HEAD, 
    NODE_COUNT, 
    None
  )
  save_weight_logs(initial_node_weights, 0, 'fedma')
  save_weight_logs(initial_node_weights, 0, 'mfedma')
  fedAVG_train_loss_history.append(initial_node_losses)
  matched_fedAVG_train_loss_history.append(initial_node_losses)

  # Run experiments on FedAVG aggregation method
  fedAVG_weights = calculate_federated_weights(initial_node_weights[0], initial_node_weights[1])
  fedAVG_test_loss_history.append(test_a_model(test_input_seqs, test_label_seqs, fedAVG_weights, D_MODEL, ATTENTION_HEAD))
  for i in range(COMMUNICATION_ROUNDS):
    node_losses, node_weights = perform_1_federated_training_round(
      input_seqs, 
      label_seqs, 
      D_MODEL, 
      ATTENTION_HEAD, 
      NODE_COUNT, 
      fedAVG_weights
    )
    save_weight_logs(node_weights, i + 1, 'fedma')
    fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
    fedAVG_train_loss_history.append(node_losses)
    fedAVG_test_loss_history.append(test_a_model(test_input_seqs, test_label_seqs, fedAVG_weights, D_MODEL, ATTENTION_HEAD))

  # Run experiments on Matched FedAVG aggregation method
  node_weights = perform_weight_permutation_matching(initial_node_weights, D_MODEL, ATTENTION_HEAD)
  matched_fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
  matched_fedAVG_test_loss_history.append(test_a_model(test_input_seqs, test_label_seqs, matched_fedAVG_weights, D_MODEL, ATTENTION_HEAD))
  for i in range(COMMUNICATION_ROUNDS):
    node_losses, node_weights = perform_1_federated_training_round(
      input_seqs, 
      label_seqs, 
      D_MODEL, 
      ATTENTION_HEAD, 
      NODE_COUNT, 
      matched_fedAVG_weights
    )
    save_weight_logs(node_weights, i + 1, 'mfedma')
    # Calculate the best permutation matrix to match the weights from 2 nodes
    node_weights = perform_weight_permutation_matching(node_weights, D_MODEL, ATTENTION_HEAD)
    matched_fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
    matched_fedAVG_train_loss_history.append(node_losses)
    matched_fedAVG_test_loss_history.append(test_a_model(test_input_seqs, test_label_seqs, matched_fedAVG_weights, D_MODEL, ATTENTION_HEAD))

  # Print experiemental results
  for i in range(COMMUNICATION_ROUNDS):
    train_loss = fedAVG_train_loss_history[i]
    test_loss = fedAVG_test_loss_history[i]
    print('FedAVG, round: ' + str(i) + ', Train Loss: ' + str(train_loss)+ ', Test Loss: ' + str(test_loss))
  for i in range(COMMUNICATION_ROUNDS):
    train_loss = matched_fedAVG_train_loss_history[i]
    test_loss = matched_fedAVG_test_loss_history[i]
    print('Matched FedAVG, round: ' + str(i) + ', Train Loss: ' + str(train_loss)+ ', Test Loss: ' + str(test_loss))

  # Save output to log file
  with open('6_output.csv', 'w', encoding='utf-8') as fout:
    fout.write('Federated Round,FedAVG Local Loss 1,FedAVG Local Loss 2,Matched FedAVG Local Loss 1,Matched FedAVG Local Loss 2,FedAVG Global Loss,Matched FedAVG Global Loss\n')
    for i in range(COMMUNICATION_ROUNDS):
      fedAVG_train_loss = fedAVG_train_loss_history[i]
      fedAVG_test_loss = fedAVG_test_loss_history[i]
      matched_fedAVG_train_loss = matched_fedAVG_train_loss_history[i]
      matched_fedAVG_test_loss = matched_fedAVG_test_loss_history[i]
      fout.write(
        str(i) + ',' + 
        str(fedAVG_train_loss[0]) + ',' + 
        str(fedAVG_train_loss[1]) + ',' + 
        str(matched_fedAVG_train_loss[0]) + ',' + 
        str(matched_fedAVG_train_loss[1]) + ',' + 
        str(fedAVG_test_loss) + ',' + 
        str(matched_fedAVG_test_loss) + 
        '\n')

  print('Finished.')
