import numpy as np

PERM_SIZE = 3

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

xx = generate_permutaion_matrix(PERM_SIZE)
print(xx)
for x in xx:
  print(x)