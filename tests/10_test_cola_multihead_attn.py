# This unit test case is for tesing single layer multi-head attention with CoLA dataset.
# This model is like this: [INPUT (input_ids)] => [EMBEDDING] => [MULTI-HEAD ATTENTION] => [DENSE] => [OUTPUT (binary)]
# For simpicity, we use basic SGD as the optimizer.
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

# Model configuration
USE_POSITIONAL_ENCODING = True

# Training Parameters
COMMUNICATION_ROUNDS = 10
LOCAL_TRAIN_EPOCH = 10
ATTENTION_HEAD = 4
BATCH_SIZE = 5
BATCH_NUM = -1
D_MODEL = 16
SEQ_LEN = -1 # -1 For automatically detected from training data maximum length
VOCAB_SIZE = 150

# Number of federated nodes
NODE_COUNT = 2

# String speical token specifications
TOKEN_UNKNOWN = 1
TOKEN_CLS = 2
TOKEN_SEP = 3

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

#################################################################
# FUNCTIONS FOR LOADING COLA DATASET
def check_and_download_cola():
  data_folder = os.path.join('dataset', 'cola')
  if not os.path.exists(data_folder):
    print('[INFO] No data folder found, recreating...')
    os.makedirs(data_folder)
  zip_filename = 'cola_public_1.1.zip'
  zip_filepath = os.path.join(data_folder, zip_filename)
  if not os.path.exists(zip_filepath):
    print('[INFO] No zip file found, downloading...')
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
    r = requests.get(url, allow_redirects=True)
    open(zip_filepath, 'wb').write(r.content)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
      zip_ref.extractall(data_folder)

def read_cola_data_file(file_path):
  data = []
  with open(file_path, 'r', encoding='utf-8') as fin:
    for line in fin:
      columns = line.split('\t')
      data_row = {
        'id': columns[0].strip(),
        'label': columns[1].strip(),
        'input': columns[3].strip()
      }
      print(data_row['input'] + ' => ' + data_row['label'])
      data.append(data_row)
  return data

def load_cola_data():
  check_and_download_cola()
  data_path_train = os.path.join('dataset', 'cola', 'cola_public', 'raw', 'in_domain_train.tsv')
  data_path_dev = os.path.join('dataset', 'cola', 'cola_public', 'raw', 'in_domain_dev.tsv')
  data_train = read_cola_data_file(data_path_train)
  data_dev = read_cola_data_file(data_path_dev)
  return data_train, data_dev

def load_encoded_cola_data():
  encoded_data_train_path = os.path.join('dataset', 'cola', 'train.pk')
  encoded_data_dev_path = os.path.join('dataset', 'cola', 'dev.pk')

  data_train = None
  data_dev = None

  if os.path.exists(encoded_data_train_path) and os.path.exists(encoded_data_dev_path):
    # Load from processed file
    print('[INFO] Loading data from pre-generated file.')
    with open(encoded_data_train_path,'rb') as fin:
      data_train = pickle.load(fin)
    with open(encoded_data_dev_path,'rb') as fin:
      data_dev = pickle.load(fin)

    return data_train, data_dev

  data_train, data_dev = load_cola_data()
  max_dict_size = VOCAB_SIZE
  sentence_piece_processor = spm.SentencePieceProcessor()
  print('[INFO] Max Dictionary Size = ' + str(max_dict_size))
  dict_vocab_path = os.path.join('dataset', 'cola', 'spm.vocab')
  dict_model_path = os.path.join('dataset', 'cola', 'spm.model')

  if not os.path.exists(dict_model_path):
    print('[INFO] No SPM model file, creating...')
    # Create raw corpus file to train SPM
    raw_corpus_file = os.path.join('dataset', 'cola', 'corpus.txt')
    with open(raw_corpus_file, 'w', encoding='utf-8') as fout:
      for data_row in data_train:
        fout.write(data_row['input'] + '\n')
      
    # Train sentence piece model
    spm.SentencePieceTrainer.Train('--pad_id=0 --bos_id=' + str(TOKEN_CLS) + 
      ' --eos_id=' + str(TOKEN_SEP) + 
      ' --unk_id=' + str(TOKEN_UNKNOWN) + 
      ' --user_defined_symbols=<MASK> --input=' + 
      raw_corpus_file + 
      ' --model_prefix=sp --vocab_size=' + str(max_dict_size) + ' --hard_vocab_limit=false')

    # Delete raw corpus file
    os.remove(raw_corpus_file)

    # Move sp.model / sp.vocab to the dict paths
    os.rename("sp.model", dict_model_path)
    os.rename("sp.vocab", dict_vocab_path)

    sentence_piece_processor.Load(dict_model_path)            
  else:
    sentence_piece_processor.Load(dict_model_path)

  print('[INFO] Dictionary size = ' +str(sentence_piece_processor.GetPieceSize()))

  # Perform encoding
  for data in data_train:
    encoded_data = sentence_piece_processor.EncodeAsIds(data['input'])
    data['input_ids'] = encoded_data
  for data in data_dev:
    encoded_data = sentence_piece_processor.EncodeAsIds(data['input'])
    data['input_ids'] = encoded_data

  # Save pre-generated file
  with open(encoded_data_train_path, 'wb') as fout:
    pickle.dump(data_train, fout)
  with open(encoded_data_dev_path, 'wb') as fout:
    pickle.dump(data_dev, fout)

  return data_train, data_dev

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
  input_tensor = tf.placeholder(shape=(batch, seq_len), dtype=tf.int32)
  mask_tensor = tf.placeholder(shape=(batch, seq_len), dtype=tf.float32)

  # Perform embedding from out-hot id into d_model dimension
  input_ids = input_tensor
  print(input_ids)
  with tf.variable_scope('word_embedding', reuse=False):
    embedding_table = tf.get_variable(
        name='kernel',
        shape=[vocab_size, d_model]
      )
  input_ids = tf.nn.embedding_lookup(embedding_table, input_ids)

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

  # Pooled output is the hidden state of the 1st token
  pooled_output_tensor = output_tensor[:, 0]

  # Add binary classification layers
  prediction_tensor = tf.layers.dense(pooled_output_tensor, 1, name='prediction')
  logprob_tensor = tf.nn.sigmoid(prediction_tensor, name ='sigmoid')

  return (input_tensor, mask_tensor, prediction_tensor, disagreement_cost, logprob_tensor)
     
# Build loss graph to evaluate the model
def build_loss_graph(output_tensor, batch, seq_len, d_model, additional_costs):
  label_tensor = tf.placeholder(shape=output_tensor.get_shape(), dtype=tf.float32)
  classification_losses = tf.losses.sigmoid_cross_entropy(label_tensor, output_tensor)
  total_loss = classification_losses + tf.reduce_mean(additional_costs)
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
  with tf.variable_scope('prediction', reuse=True):
    prediction_kernel = sess.run(tf.get_variable('kernel'))
    prediction_bias = sess.run(tf.get_variable('bias'))
  with tf.variable_scope('word_embedding', reuse=True):
    embedding_kernel = sess.run(tf.get_variable('kernel'))
  return [K_kernel, K_bias, Q_kernel, Q_bias, V_kernel, V_bias, output_kernel, output_bias, prediction_kernel, prediction_bias, embedding_kernel]

# Set all model weights to current graph
def set_all_variables(sess, var_list):
  [K_kernel, K_bias, Q_kernel, Q_bias, V_kernel, V_bias, output_kernel, output_bias, prediction_kernel, prediction_bias, embedding_kernel] = var_list
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
  with tf.variable_scope('prediction', reuse=True):
    sess.run(tf.get_variable('kernel').assign(prediction_kernel))
    sess.run(tf.get_variable('bias').assign(prediction_bias))
  with tf.variable_scope('word_embedding', reuse=True):
    sess.run(tf.get_variable('kernel').assign(embedding_kernel))

# Run an evaluation on a model initialized with the specified weights
# Output of the model will be all weights of the trained model
def test_a_model(input_seq, mask_seq, label_seq, var_list, d_model, head, print_output=True):
  # Clear all stuffs in default graph, so we can start fresh
  tf.reset_default_graph()
  # We want each session to have different random seed, but we need each run to have the same random sequence
  tf.set_random_seed(random.randint(0, 65535))

  batch_size = len(input_seq[0])
  seq_len = len(input_seq[0][0])

  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  (input_tensor, mask_tensor, output_tensor, disagreement_cost, logprob_tensor) = build_model(batch=batch_size, seq_len=seq_len, vocab_size=VOCAB_SIZE, d_model=d_model, head=head)
  (label_tensor, loss, classification_loss) = build_loss_graph(output_tensor=output_tensor, batch=batch_size, seq_len=seq_len, d_model=d_model, additional_costs=[disagreement_cost])
  sess.run(tf.global_variables_initializer())
  set_all_variables(sess, var_list)

  avg_loss = 0.0
  avg_disgreement_loss = 0.0
  avg_classification_loss = 0.0
  avg_accuracy = 0.0
  for input_sample, mask_sample, label_sample in zip(input_seq, mask_seq, label_seq):
    [output_vals, loss_vals, disagreement_cost_vals, classification_loss_vals, logprob_vals] = sess.run([output_tensor, loss, disagreement_cost, classification_loss, logprob_tensor], feed_dict={input_tensor: input_sample, mask_tensor: mask_sample, label_tensor: label_sample})
    avg_loss = avg_loss + loss_vals
    avg_disgreement_loss = avg_disgreement_loss + disagreement_cost_vals
    avg_classification_loss = avg_classification_loss + classification_loss_vals
    labels = np.array(label_sample)
    predictions = (logprob_vals >= 0.5).astype(int)
    scores = (predictions == labels).astype(int)
    scores = np.average(scores)
    avg_accuracy = avg_accuracy + scores
  avg_loss = avg_loss / len(input_seq)
  avg_disgreement_loss = avg_disgreement_loss / len(input_seq)
  avg_classification_loss = avg_classification_loss / len(input_seq)
  avg_accuracy = avg_accuracy / len(input_seq)

  if print_output:
    print('===== TEST STAT =====')
    '''
    print('=== Input Values ===')
    print(input_seq)
    print('=== Label Values ===')
    print(label_seq)
    print('=== Output Values ===')
    print(output_vals)
    '''
    print('=== Loss Values ===')
    print(avg_loss)
    print('=== Classification Loss Values ===')
    print(avg_classification_loss)
    '''
    print('=== Disagreement Loss Values ===')
    print(avg_disgreement_loss)
    '''
    print('=== Accuracy ===')
    print(avg_accuracy)
  return avg_loss, avg_disgreement_loss, avg_classification_loss, avg_accuracy

# Run an experiment by initial new model and perform training for 10 steps.
# Output of the model will be all weights of the trained model
def train_a_model(input_seq, mask_seq, label_seq, vocab_size, d_model, head, init_weights, print_output=True):
  # Clear all stuffs in default graph, so we can start fresh
  tf.reset_default_graph()
  # We want each session to have different random seed, but we need each run to have the same random sequence
  tf.set_random_seed(random.randint(0, 65535))

  batch_size = len(input_seq[0])
  seq_len = len(input_seq[0][0])

  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)
  (input_tensor, mask_tensor, output_tensor, disagreement_cost, logprob_tensor) = build_model(batch=batch_size, seq_len=seq_len, vocab_size=vocab_size, d_model=d_model, head=head)
  (label_tensor, train_op, loss, classification_loss) = build_train_graph(output_tensor=output_tensor, batch=batch_size, seq_len=seq_len, d_model=d_model, additional_costs=[disagreement_cost])
  sess.run(tf.global_variables_initializer())

  if init_weights is not None:
    set_all_variables(sess, init_weights)

  for i in range(LOCAL_TRAIN_EPOCH):
    if print_output:
      print('EPOCH: ' + str(i))

    avg_loss = 0.0
    avg_disagreement_loss = 0.0
    avg_classification_loss = 0.0
    avg_accuracy = 0.0
    for input_sample, mask_sample, label_sample in zip(input_seq, mask_seq, label_seq):
      [output_vals, loss_vals, disagreement_cost_vals, classification_loss_vals, logprob_vals, _] = sess.run([output_tensor, loss, disagreement_cost, classification_loss, logprob_tensor, train_op], feed_dict={input_tensor: input_sample, mask_tensor: mask_sample, label_tensor: label_sample})
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
    print('===== TRAIN STAT =====')
    '''
    print('=== Input Values ===')
    print(input_seq)
    print('=== Label Values ===')
    print(label_seq)
    print('=== Output Values ===')
    print(output_vals)
    '''
    print('=== Loss Values ===')
    print(avg_loss)
    print('=== Classification Loss Values ===')
    print(avg_classification_loss)
    '''
    print('=== Disagreement Loss Values ===')
    print(avg_disagreement_loss)
    '''
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
    var_list[8], # prediction_kernel (no permutation needed)
    var_list[9], # prediction_bias (no permutation needed)
    var_list[10], # word_embedding_kernel (no permutation needed)
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
    non_perm_list[1], # prediction_kernel (no permutation needed)
    non_perm_list[2], # prediction_bias (no permutation needed)
    non_perm_list[3], # word_embedding_kernel (no permutation needed)
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

def truncate_and_pad(ar, target_len):
  # Add [CLS] and [SEP] token infront and after input sequence respectively
  target_len = target_len + 2
  ret = []
  mask = []
  ret.append(TOKEN_CLS)
  mask.append(1.0)
  for tok in ar:
    ret.append(tok)
    mask.append(1.0)
  ret.append(TOKEN_SEP)
  mask.append(1.0)
  ret = ret[0:target_len]
  mask = mask[0:target_len]
  while len(ret) < target_len:
    ret.append(0)
    mask.append(0.0)
  return ret, mask

# Function to generate training data batches, using CoLA dataset
def simulate_federated_data(batch_size, batch_num, seq_len, dataset, node_count):
  input_seqs = []
  mask_seqs = []
  label_seqs = []

  all_training_rows = len(dataset)
  node_training_rows = all_training_rows // node_count
  print('All training data count = ' + str(all_training_rows) + ', each node get = ' + str(node_training_rows))

  for i in range(node_count):
    input_batches = []
    mask_batches = []
    label_batches = []

    start_idx = node_training_rows * i
    batch_count = node_training_rows // batch_size
    if batch_num != -1:
      batch_count = min(batch_count, batch_num) # Limit max batch count
    for j in range(batch_count):
      idx = start_idx + j * batch_size
      batch = dataset[idx: idx + batch_size]
      input_ids_batch = [a['input_ids'] for a in batch]
      input_batch = []
      mask_batch = []
      for input_ids in input_ids_batch:
        ids, masks = truncate_and_pad(input_ids, seq_len)
        input_batch.append(ids)
        mask_batch.append(masks)
      label_batch = [[int(a['label'])] for a in batch] # We need 2D array even for binary label
      input_batches.append(input_batch)
      mask_batches.append(mask_batch)
      label_batches.append(label_batch)

    input_seqs.append(input_batches)
    mask_seqs.append(mask_batches)
    label_seqs.append(label_batches)

  return input_seqs, mask_seqs, label_seqs

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
  file_path = os.path.join('weight_logs', '10_benchmark_' + algor + '_' + str(epoch) + '.pkl')
  with open(file_path, 'wb') as fout:
    pickle.dump(node_weights, fout)

# Load CoLA dataset
data_train, data_dev = load_encoded_cola_data()
print('Training data has ' + str(len(data_train)) + ' rows.')
print('Validation data has ' + str(len(data_dev)) + ' rows.')

# Find max length of training data
max_len = 0
for data in data_train:
  if max_len < len(data['input_ids']):
    max_len = len(data['input_ids'])
print('[INFO] Max training data seq_len = ' + str(max_len))

# Override if we are forcing max length here
if SEQ_LEN != -1:
  max_len = SEQ_LEN

# Simulate input and label.
# Both federated node see training data from different distribution of X
input_seqs, mask_seqs, label_seqs = simulate_federated_data(batch_size=BATCH_SIZE, 
  batch_num=BATCH_NUM, 
  seq_len=max_len,
  dataset=data_train, 
  node_count=NODE_COUNT
)

test_input_seqs, test_mask_seqs, test_label_seqs = simulate_federated_data(batch_size=BATCH_SIZE, 
  batch_num=BATCH_NUM, 
  seq_len=max_len,
  dataset=data_dev,
  node_count=1 # We use single central node to do validation
)
# Dev dataset has only single node
test_input_seqs = test_input_seqs[0]
test_mask_seqs = test_mask_seqs[0]
test_label_seqs = test_label_seqs[0]

for i in range(NODE_COUNT):  
  print('-------------------------------------------')
  print('Local training data for Federated Node: ' + str(i))
  print('-------------------------------------------')
  print('input_seq: X[0] has length of = ' + str(len(input_seqs[i])))

'''
print('-------------------------------------------')
print('Global test data')
print('-------------------------------------------')
print('input_seq: X[0] has average values = ' + str(np.average(np.array(test_input_seqs)[:,0,:,0])))
'''

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
avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, fedAVG_weights, D_MODEL, ATTENTION_HEAD)
fedAVG_test_loss_history.append(avg_loss)
fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
fedAVG_test_classification_loss_history.append(avg_classification_loss)
fedAVG_test_accuracy_history.append(avg_accuracy)
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
  avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, fedAVG_weights, D_MODEL, ATTENTION_HEAD)
  fedAVG_test_loss_history.append(avg_loss)
  fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
  fedAVG_test_classification_loss_history.append(avg_classification_loss)
  fedAVG_test_accuracy_history.append(avg_accuracy)

# Run experiments on Matched FedAVG aggregation method
node_weights = perform_weight_permutation_matching(initial_node_weights, D_MODEL, ATTENTION_HEAD)
matched_fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, matched_fedAVG_weights, D_MODEL, ATTENTION_HEAD)
matched_fedAVG_test_loss_history.append(avg_loss)
matched_fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
matched_fedAVG_test_classification_loss_history.append(avg_classification_loss)
matched_fedAVG_test_accuracy_history.append(avg_accuracy)
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
  node_weights = perform_weight_permutation_matching(node_weights, D_MODEL, ATTENTION_HEAD)
  matched_fedAVG_weights = calculate_federated_weights(node_weights[0], node_weights[1])
  matched_fedAVG_train_loss_history.append(node_losses)
  matched_fedAVG_train_disagreement_loss_history.append(node_disagreement_losses)
  matched_fedAVG_train_classification_loss_history.append(node_classification_losses)
  matched_fedAVG_train_accuracy_history.append(node_accuracy)
  avg_loss, avg_disagreement_loss, avg_classification_loss, avg_accuracy = test_a_model(test_input_seqs, test_mask_seqs, test_label_seqs, matched_fedAVG_weights, D_MODEL, ATTENTION_HEAD)
  matched_fedAVG_test_loss_history.append(avg_loss)
  matched_fedAVG_test_disagreement_loss_history.append(avg_disagreement_loss)
  matched_fedAVG_test_classification_loss_history.append(avg_classification_loss)
  matched_fedAVG_test_accuracy_history.append(avg_accuracy)

# Print experiemental results
for i in range(COMMUNICATION_ROUNDS):
  train_loss = fedAVG_train_loss_history[i]
  test_loss = fedAVG_test_loss_history[i]
  train_acc = fedAVG_train_accuracy_history[i]
  test_acc = fedAVG_test_accuracy_history[i]
  print('FedAVG, round: ' + str(i) + ', Train Loss: ' + str(train_loss)+ ', Test Loss: ' + str(test_loss) + ', Train Acc: ' + str(train_acc) + ', Test Acc: ' + str(test_acc))
for i in range(COMMUNICATION_ROUNDS):
  train_loss = matched_fedAVG_train_loss_history[i]
  test_loss = matched_fedAVG_test_loss_history[i]
  train_acc = matched_fedAVG_train_accuracy_history[i]
  test_acc = matched_fedAVG_test_accuracy_history[i]
  print('Matched FedAVG, round: ' + str(i) + ', Train Loss: ' + str(train_loss)+ ', Test Loss: ' + str(test_loss) + ', Train Acc: ' + str(train_acc) + ', Test Acc: ' + str(test_acc))

# Save output to log file
with open('10_output.csv', 'w', encoding='utf-8') as fout:
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
  for i in range(COMMUNICATION_ROUNDS):
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

print('Finished.')

