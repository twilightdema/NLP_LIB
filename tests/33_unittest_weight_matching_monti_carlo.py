# This unit test is for verifying implementation of Monti-Carlo method for weight matching.
# The goal is to verify that, given either Euclidian or Cosine distance function as a cost function,
# performing update in Monti-Carlo manner can minimize the cost function like in KNN optimization alorithm.
# If we can verify this, then we can use this method to perform matching for N nodes.

import numpy as np
import tensorflow.compat.v1 as tf
import random
import os
import sys

# Number of nodes.
NODE_COUNT = 10

# Dimension of weight in each node.
WEIGHT_DIMENSION = (2, 5)

# Number of weight parameters (In Multi-Head Attention, it is equal to number of head)
ATTENTION_HEAD = 4

# Maximum iteration of monti-carlo update allowed.
MAX_MONTI_CARLO_ITERATION = 2000

# Min loss progress, any loss value change less than this will trigger termination of monti-carlo iteration.
MIN_LOSS_PROGRESS = 0.01

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
    distance = distance + distance_func(permutated_node_weights, global_weights)

  return distance

def perform_monti_carlo_weight_matching(weights_list, permutation_matrics, distance_func, iteration_count, min_delta):
  current_delta = sys.float_info.max

  for i in range(iteration_count):
    print('Monti-Carlo Iteration: ' + str(i))

def simulate_weights(node_count, head_count, dimension):
  weights = np.random.rand(node_count, head_count, *dimension) * 2.0 - 1.0
  return weights

# Random weights
weights_list = simulate_weights(NODE_COUNT, ATTENTION_HEAD, WEIGHT_DIMENSION)

# Initialize permutation matrics
permutation_matrics = [[a for a in range(ATTENTION_HEAD)][::-1] for _ in range(NODE_COUNT)]

# Initial total loss
loss = total_loss(weights_list, permutation_matrics, distance_function_euclidian)
print('INITIAL LOSS:')
print(loss)

# Perform optimization
# perform_monti_carlo_weight_matching(weights_list, permutation_matrics, distance_function_euclidian, MAX_MONTI_CARLO_ITERATION, MIN_LOSS_PROGRESS)

# Final total loss
print('FINAL LOSS:')
loss = total_loss(weights_list, permutation_matrics, distance_function_euclidian)

print(loss)