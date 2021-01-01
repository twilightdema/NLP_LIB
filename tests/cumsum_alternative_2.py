import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K

import tensorflow.compat.v1 as tf

def shape_list(x):
  """Return list of dims, statically where possible."""
  x = tf.convert_to_tensor(x)
  # If unknown rank, return dynamic shape
  if x.get_shape().dims is None:
    return tf.shape(x)
  static = x.get_shape().as_list()
  shape = tf.shape(x)
  ret = []
  for i in range(len(static)):
    dim = static[i]
    if dim is None:
      dim = shape[i]
    ret.append(dim)
  return ret

x_val = [[1,7,6,8,0,0],[6,7,5,0,0,0]]
x = tf.constant(x_val)

len_s = tf.shape(x)[1]
bs = tf.shape(x)[:1]

mask = K.cumsum(tf.eye(len_s, batch_shape=bs), 1)
with tf.Session(config = tf.ConfigProto(log_device_placement = True, allow_soft_placement=False)) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  mask_val = sess.run(mask)
  print('Result from K.cumsum (No-GPU supported) = ' + str(mask_val))

ones_part = tf.ones([bs[0], len_s, len_s])
mask = tf.linalg.band_part(ones_part, -1, 0)
with tf.Session(config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=False)) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  mask_val = sess.run(mask)
  print('Result from K.range (GPU supported) ' + str(mask_val))
