# This unit test is for finding out if the monti-carlo algorithm for N-Node attention heads matching
# (in Experiment 33) can always find the "best" solution for matching problem or not.
# Our assumption is that the monti-carlo should be "approximation" algorithm, like K-Mean clustering.
# However, it is not exactly noted in many paper using it for neuron matching (in PFMN and FedMA).
# So we want to proof it by perform 10 trials on comparison between exact matching and monti-carlo matching.
# Because exact matching is costly, we will try to use 3 Nodes, 4 Heads as our test cases.
# Note that more than 4 Heads can be implemented using Hungarian algorithm to speed up factorial time to polynomial time,
# but because it produces the same exact optimal solution as brute force method, we choose not to implement it now.
import numpy as np
import tensorflow.compat.v1 as tf
import random
import os
import sys

# Number of trial to perform the comparison
TRIAL_NUM = 10

# Number of nodes.
NODE_COUNT = 3

# Dimension of weight in each node.
WEIGHT_DIMENSION = (128, 128)

# Number of weight parameters (In Multi-Head Attention, it is equal to number of head)
ATTENTION_HEAD = 4

# Maximum iteration of monti-carlo update allowed.
MAX_MONTI_CARLO_ITERATION = 100

# Min loss progress, any loss value change less than this will trigger termination of monti-carlo iteration.
MIN_LOSS_PROGRESS = 0.000001

####################################################################
# FUNCTION FOR SETUP RANDOMSEED SO THAT EXPERIMENTS ARE REPRODUCIBLE
RANDOM_SEED = 4567
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

# Function for generating all possible permulation matrix for perm_size dimension
# Using generator pattern for sage of performance and memory efficiency.
def generate_permutaion_matrix(perm_size):
  current = [i for i in range(perm_size)]
  def gen_perm(pos):
    if pos == len(current):
      yield np.array(current)
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

def distance_function_euclidian(list1, list2):
  acc_dist = 0.0
  for a, b in zip(list1, list2):
    acc_dist = acc_dist + np.sum(np.abs(a - b))
  return acc_dist

def distance_function_cosine(list1, list2):
  acc_dist = 0.0
  for a, b in zip(list1, list2):
    cos_dist = np.inner(a.flatten(), b.flatten())
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    acc_dist = acc_dist + cos_dist / norm
  acc_dist = -acc_dist
  return acc_dist

# Function to calculate expected global weights for each head.
# Expected global weight is at the central of cluster for each head.
def expected_global_weights(weights_list, permutation_matrics):
  node_count = weights_list.shape[0]
  head_count = weights_list.shape[1]

  global_weights = None
  for node_weights, permutation_matrix in zip(weights_list, permutation_matrics):
    permutated_node_weights = node_weights[permutation_matrix]
    if global_weights is None:
      global_weights = np.copy(permutated_node_weights)
    else:
      global_weights = global_weights + permutated_node_weights

  global_weights = global_weights / node_count
  return global_weights

# Calculate current total loss of permutation matrices
def total_loss(weights_list, permutation_matrics, distance_func):
  node_count = weights_list.shape[0]
  head_count = weights_list.shape[1]

  # Calculate central of cluster for each head to be expected value of global node
  global_weights = expected_global_weights(weights_list, permutation_matrics)

  # Calculate total distance to the expected global node
  distance = 0.0
  for node_weights, permutation_matrix in zip(weights_list, permutation_matrics):
    permutated_node_weights = node_weights[permutation_matrix]
    distance = distance + distance_func([permutated_node_weights], [global_weights])

  return distance

# Function for finding best permutaion matrix for current node compared to global node
# Note that this process can be parallelized in Map-Reduce style (probably in GPU!).
def find_best_permutation_matrix(this_node_weights, global_weights, distanc_func):
  head_count = this_node_weights.shape[0]
  perm_mats = generate_permutaion_matrix(head_count)
  min_distance = sys.float_info.max
  min_perm_mat = None
  for perm_mat in perm_mats:
    permutated_node_weights = this_node_weights[perm_mat]
    distance = distanc_func([permutated_node_weights], [global_weights])
    # print(' - Perm Mat: ' + str(perm_mat) + ', Distance: ' + str(distance))
    if distance < min_distance:
      min_distance = distance
      min_perm_mat = perm_mat
  return min_perm_mat, min_distance

def perform_monti_carlo_weight_matching(weights_list, permutation_matrics, distance_func, iteration_count, min_delta):
  previous_loss = sys.float_info.max
  node_count = weights_list.shape[0]

  for i in range(iteration_count):
    # print('Monti-Carlo Iteration: ' + str(i))
    for n in range(node_count):
      #print(' Expectation Maximization Node: ' + str(n))
      this_node_weights = weights_list[n]
      this_permutation_matrx = permutation_matrics[n]
      remaining_node_weights_list = np.concatenate((weights_list[:n], weights_list[n+1:]))
      remaining_permutation_matrics = np.concatenate((permutation_matrics[:n], permutation_matrics[n+1:]))

      global_weights = expected_global_weights(remaining_node_weights_list, remaining_permutation_matrics)
      min_perm_mat, min_distance = find_best_permutation_matrix(this_node_weights, global_weights, distance_func)
      #print('  Min Permutaion Matrtix = ' + str(min_perm_mat))
      #print('  Min Permutation Cost = ' + str(min_distance))

      permutation_matrics[n] = min_perm_mat

    # Debug Log
    '''
    for i, permutation_matrix in enumerate(permutation_matrics):
      print('Perm Mat of Node: ' + str(i) + ' = ' + str(permutation_matrix))
    '''

    loss = total_loss(weights_list, permutation_matrics, distance_func)
    # print('Iteration: ' + str(i) + ', total loss: ' + str(loss))
    delta = abs(loss - previous_loss)
    if delta < min_delta:
      # print('Too low Delta: ' + str(delta) + '. Stop opimizaion.')
      break
    previous_loss = loss

# Function to find best permutation matrix for N-Node using exact method. This one is slow.
_exact_min_distance = sys.maxsize
_exact_min_perm_mats = None
def perform_exact_weight_matching(weights_list, distance_func):
  global _exact_min_distance
  global _exact_min_perm_mats
  node_count = weights_list.shape[0]
  head_count = weights_list.shape[1]

  _exact_min_distance = sys.maxsize
  _exact_min_perm_mats = None

  def recur_(weights_list, i, perm_mat_list):
    global _exact_min_distance
    global _exact_min_perm_mats
    if i >= len(weights_list):
      loss = total_loss(weights_list, perm_mat_list, distance_func)
      if loss < _exact_min_distance:
        _exact_min_distance = loss
        _exact_min_perm_mats = np.copy(perm_mat_list)
      return

    possible_perm_mat = list(generate_permutaion_matrix(head_count))
    # print(possible_perm_mat)
    for perm_mat in possible_perm_mat:
      perm_mat_list.append(np.copy(perm_mat))
      recur_(weights_list, i+1, perm_mat_list)
      perm_mat_list.pop()

  recur_(weights_list, 0, [])
  return _exact_min_distance, _exact_min_perm_mats

def simulate_weights(node_count, head_count, dimension):
  weights = np.random.rand(node_count, head_count, *dimension) * 2.0 - 1.0
  return weights

# Random weights
weights_list = simulate_weights(NODE_COUNT, ATTENTION_HEAD, WEIGHT_DIMENSION)

# Specify Distance Function
distance_func = distance_function_euclidian

for trial in range(TRIAL_NUM):
  # Initialize permutation matrics, we randomly initialize it so each trial starts from different initial assignments.
  permutation_matrics = []
  for i in range(NODE_COUNT):
    remainings = [a for a in range(ATTENTION_HEAD)]
    perm_mat = []
    for j in range(ATTENTION_HEAD):
      idx = random.randint(0, ATTENTION_HEAD-1-j)
      perm_mat.append(remainings.pop(idx))
    permutation_matrics.append(perm_mat)
    #print('Perm Mat of Node: ' + str(i) + ' = ' + str(perm_mat))
  permutation_matrics = np.array(permutation_matrics)

  # Initial total loss
  initial_permutation_matrics = np.copy(permutation_matrics)
  initial_loss = total_loss(weights_list, permutation_matrics, distance_func)

  # Perform optimization
  #print('Perform Optimization')
  perform_monti_carlo_weight_matching(weights_list, permutation_matrics, distance_func, MAX_MONTI_CARLO_ITERATION, MIN_LOSS_PROGRESS)

  # Find exact minimal loss using brute force method
  exact_min_loss, exact_min_perm_mats = perform_exact_weight_matching(weights_list, distance_func)

  # Final permutation matrics
  #for i, permutation_matrix in enumerate(permutation_matrics):
  #  print('Perm Mat of Node: ' + str(i) + ' = ' + str(permutation_matrix))

  # Final total loss
  loss = total_loss(weights_list, permutation_matrics, distance_func)
  loss_exact = total_loss(weights_list, exact_min_perm_mats, distance_func)
  print('Trial: ' + str(trial) + ', Exact: ' + str(loss_exact) + ' Monti-Carlo: ' + str(loss))
