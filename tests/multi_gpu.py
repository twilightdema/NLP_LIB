# Thistest is for running baseline model in Multi-GPU
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib
import tensorflow.compat.v1 as tf

# getting the number of GPUs 
def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']
num_gpu = len(get_available_gpus())

print('num_gpu = ' + str(num_gpu))

input_tensor = Input(shape=(32,))
output_tensor = Dense(32)(input_tensor)

with tf.device('/cpu'):
  model = Model(input_tensor, output_tensor)
  mg_model = multi_gpu_model(model, gpus=num_gpu)
  mg_model.compile(optimizer='adam', loss='categorical_crossentropy')
  mg_model.summary()

with tf.Session(config = tf.ConfigProto(log_device_placement = True, allow_soft_placement=False)) as sess:
  init = tf.global_variables_initializer()
  sess.run(init) 
  print("--------------------------------")
  print("--------------------------------")
  print("--------------------------------")
  print("--------------------------------")
  print("--------------------------------")
  print("--------------------------------")

  x = np.random.random((32000, 32))
  y = np.random.random((32000, 32))
  mg_model.fit(x, y, epochs=100, batch_size=100)

