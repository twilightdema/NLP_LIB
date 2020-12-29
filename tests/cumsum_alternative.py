import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
import tensorflow.compat.v1 as tf
from tensorflow.keras import backend as K

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
pos = K.cumsum(K.ones_like(x, 'int32'), 1)
with tf.Session(config = tf.ConfigProto(log_device_placement = True, allow_soft_placement=False)) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  pos_val = sess.run(pos)
  print('Result from K.cumsum (No-GPU supported) = ' + str(pos_val))

tensor_shape = shape_list(x)
print('length = ' + str(tensor_shape[1]))  
pos2 = tf.add(tf.range(tensor_shape[1]), 1)
pos2 = tf.tile(pos2, [tensor_shape[0]])
pos2 = tf.reshape(pos2, tensor_shape)
with tf.Session(config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=False)) as sess:
  init = tf.global_variables_initializer()
  sess.run(init)
  pos2_val = sess.run(pos2)
  print('Result from K.range (GPU supported) ' + str(pos2_val))
