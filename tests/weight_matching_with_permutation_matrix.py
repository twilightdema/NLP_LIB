import numpy as np
import sys

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

def apply_permutation_matrix(input_list, perm_mat):
  return [input_list[i] for i in perm_mat]

def distance_function(list1, list2):
  acc_dist = 0.0
  for i in range(len(list1)):
    v1 = list1[i]
    v2 = list2[i]
    dist = abs(v1 - v2)
    acc_dist += dist
  print('Distance = ' + str(acc_dist))
  return acc_dist

def calculate_federated_weights(weight_list):
  return np.average(np.array(weight_list), axis=0)

NEURON_WEIGHTS_1 = [
  0.1, 0.2, 0.3, 0.4, 0.5
]

NEURON_WEIGHTS_2 = [
  0.54, 0.22, 0.1, 0.31, 0.11
]

PERM_SIZE = len(NEURON_WEIGHTS_1)

perm_mats = generate_permutaion_matrix(PERM_SIZE)
min_distance = 1.0e+6
min_perm_mat = None
for perm_mat in perm_mats:
  permutated_w1 = apply_permutation_matrix(NEURON_WEIGHTS_1, perm_mat)
  print(' - Matching: ' + str(permutated_w1) + ' with ' + str(NEURON_WEIGHTS_2) + ' (perm_mat = ' + str(perm_mat) + ')')
  distance = distance_function(permutated_w1, NEURON_WEIGHTS_2)
  if distance < min_distance:
    min_distance = distance
    min_perm_mat = list(np.array(perm_mat))

print('Minimum Distance = ' + str(min_distance))
print('ARGMIN Permutation Matrix = ' + str(min_perm_mat))

federated_weights = calculate_federated_weights([NEURON_WEIGHTS_1, 
  NEURON_WEIGHTS_2]
)
print('Normal Federated Weights = ' + str(federated_weights))

federated_weights = calculate_federated_weights([apply_permutation_matrix(NEURON_WEIGHTS_1, min_perm_mat), 
  NEURON_WEIGHTS_2]
)
print('Matched Federated Weights = ' + str(federated_weights))

