import numpy as np
import math
import random
from scipy.optimize import linear_sum_assignment

seed_value = 1234

# Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)

head_count = 2
weight_dimension = (10, 20)
weight_count = 3

# weight_list has size of weight_count x head_count x weight_dimension
weight_list = np.random.rand(weight_count, head_count, *weight_dimension)

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

perm_mats = list(generate_permutaion_matrix(head_count))

print(weight_list.shape)
print(perm_mats)

