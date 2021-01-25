import sys
import numpy as np
import math
import random
import time
from scipy.optimize import linear_sum_assignment

seed_value = 2345

# Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

head_count = 8
weight_dimension = (2, 4)
weight_count = 3

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
  '''
  output = []
  for input_list in perm_set:
    print(input_list)
    output_w = np.array([input_list[i] for i in perm_mat])
    print(output_w)
    output.append(output_w)
  return output
  '''
  return [np.array([input_list[i] for i in perm_mat]) for input_list in perm_set]

def distance_function_euclidian(list1, list2):
  acc_dist = 0.0
  for a, b in zip(list1, list2):
    acc_dist = acc_dist + np.sum(np.abs(a - b))
  #print('Distance = ' + str(acc_dist))
  return acc_dist

def distance_function_cosine(list1, list2):
  acc_dist = 0.0
  for a, b in zip(list1, list2):
    cos_dist = np.inner(a.flatten(), b.flatten())
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    acc_dist = acc_dist + cos_dist / norm
  acc_dist = -acc_dist
  #print('Distance = ' + str(acc_dist))
  return acc_dist

distance_function = distance_function_euclidian

def find_best_permutation_matrix(this_node_weights, global_weights, distanc_func):
  # Weights has dimension of [NUM_WEIGHT x HEAD x ...]
  head_count = len(this_node_weights[0])
  perm_mats = generate_permutaion_matrix(head_count)
  min_distance = sys.float_info.max
  min_perm_mat = None
  for perm_mat in perm_mats:
    permutated_node_weights = apply_permutation_matrix(this_node_weights, perm_mat)
    distance = distanc_func(permutated_node_weights, global_weights)
    # print(' - Perm Mat: ' + str(perm_mat) + ', Distance: ' + str(distance))
    if distance < min_distance:
      min_distance = distance
      # Make sure we copy the perm_mat to new array as the pointer is being reused in recusion call yields.
      min_perm_mat = list(np.array(perm_mat))
  print('== MIN Perm Mat: ' + str(min_perm_mat) + ', Distance: ' + str(min_distance))
  return min_perm_mat, min_distance

# Compute cost matrix for matching each head to other head across 2 nodes
def compute_cost_matrix(this_node_weights, global_weights, distanc_func):
  # Weights has dimension of [NUM_WEIGHT x HEAD x ...]
  head_count = len(this_node_weights[0])
  cost_matrix = np.zeros((head_count, head_count))
  for i in range(head_count):
    for j in range(head_count):
      # Distance between matching head i of this to head j of global
      this_head_i = [w[i] for w in this_node_weights]
      that_head_j = [w[j] for w in global_weights]
      cost_matrix[i, j] = distanc_func(this_head_i, that_head_j)
  return cost_matrix

# Function to find best permutation matrix using hungarian algorithm that run at O(N^3)
def find_best_permutation_matrix_hungarian(this_node_weights, global_weights, distanc_func):
  cost_matrix = compute_cost_matrix(this_node_weights, global_weights, distanc_func)
  row_ind, col_ind = linear_sum_assignment(cost_matrix)
  min_perm_mat = np.zeros(len(row_ind), dtype=int)
  min_perm_mat[col_ind] = row_ind
  min_perm_mat = list(min_perm_mat)
  min_distance = cost_matrix[row_ind, col_ind].sum()
  return min_perm_mat, min_distance

# weights_list has size of weight_count x head_count x weight_dimension
weights_list_local = [np.random.rand(head_count, *weight_dimension) for _ in range(weight_count)]
weights_list_global = [np.random.rand(head_count, *weight_dimension) for _ in range(weight_count)]
print('weights count = ' + str(len(weights_list_local)))
print('head count = ' + str(len(weights_list_local[0])))
print('weights dimension = ' + str(weights_list_local[0][0].shape))

print('weights_list_local = ' + str(weights_list_local))
print('weights_list_global = ' + str(weights_list_global))

print('-- Running brute force algorithm --')
start_time = time.time_ns()
min_perm_mat, min_distance = find_best_permutation_matrix(weights_list_local, weights_list_global, distance_function)
stop_time = time.time_ns()
print('min_perm_mat = ' + str(min_perm_mat))
print('min_distance = ' + str(min_distance))
print('elapse time (ns) = ' + str(stop_time - start_time))

print('-- Running hungarian algorithm --')
start_time = time.time_ns()
min_perm_mat, min_distance = find_best_permutation_matrix_hungarian(weights_list_local, weights_list_global, distance_function)
stop_time = time.time_ns()
print('min_perm_mat = ' + str(min_perm_mat))
print('min_distance = ' + str(min_distance))
print('elapse time (ns) = ' + str(stop_time - start_time))
