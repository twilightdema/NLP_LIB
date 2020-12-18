import os
import sys
import numpy as np
import tensorflow as tf
import math
import random

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
  tf.set_random_seed(seed_value)

tf.reset_default_graph()
setup_random_seed(RANDOM_SEED)

sess = tf.Session()

x = tf.constant([[1.0, 0.0, 0.2, 0.4, 0.6, 0.8]])
y = tf.layers.dense(x, 10, name='K')

sess.run(tf.global_variables_initializer())

K_kernel = None
with tf.variable_scope('K', reuse=True):
  K_kernel = sess.run(tf.get_variable('kernel'))

print(K_kernel)


