import os
import sys
import numpy as np

# This file implements functions those used for matching weights for each attention head

# Generator for purmutation matrix of dimension (perm_size x perm_size)
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

# Perform input transformation as per permutation matrix
def apply_permutation_matrix(input_list, perm_mat):
  return [input_list[i] for i in perm_mat]

# Calculate distance (matching) between each input head
def distance_function(list1, list2):
  acc_dist = 0.0
  for i in range(len(list1)):
    v1 = list1[i]
    v2 = list2[i]
    dist = abs(v1 - v2)
    acc_dist += dist
  print('Distance = ' + str(acc_dist))
  return acc_dist

'''
def calculate_federated_weights(weight_list):
  return np.average(np.array(weight_list), axis=0)
'''

# Function to find best permutation matrix for a layer of attention heads from 2 federated nodes.
# Inputs: 
#  federated_weights_list: List of weights from all nodes (shape = [node_num, weight_count])
#  transformer_layer_weights_maps: Weight index map of attention layers (shape = [layer_num, (weight_name => index_to_weight_values)])
#  n_head: Attention head count
#  layer_idx: layer to perform head matching
def match_attention_heads(federated_weights_list, transformer_layer_weights_maps, n_head, layer_idx):
  node_id_1 = 0
  node_id_2 = 1

  # Find hidden state size (Key, Query, Value) of each head
  key_size = federated_weights_list[0][transformer_layer_weights_maps[0]['electra/encoder/layer_0/attention/self/key/kernel:0']].shape
  query_size = federated_weights_list[0][transformer_layer_weights_maps[0]['electra/encoder/layer_0/attention/self/query/kernel:0']].shape
  value_size = federated_weights_list[0][transformer_layer_weights_maps[0]['electra/encoder/layer_0/attention/self/value/kernel:0']].shape
  print('[INFO]: Key size = ' + str(key_size))
  print('[INFO]: Query size = ' + str(query_size))
  print('[INFO]: Value size = ' + str(value_size))
