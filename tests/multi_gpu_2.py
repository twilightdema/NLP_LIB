# This test is for running baseline model in Multi-GPU in more complex way
# Custom optimizer and custom loss function
#
# Result from this test:
# ====================================================================================
# Custom Loss and Model with no Y in fit function (Use label_tensor from one of input)
# seems not workable in Multi-GPU. Keras besically use Single-GPU

import numpy as np
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.initializers import *
from keras.utils.training_utils import multi_gpu_model
from tensorflow.python.client import device_lib
import tensorflow as tf
from keras.optimizers import Adam

# getting the number of GPUs 
def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']
num_gpu = len(get_available_gpus())

print('num_gpu = ' + str(num_gpu))

with tf.device('/cpu'):
  input_tensor = Input(shape=(32,))
  output_tensor = Dense(32)(input_tensor)

  def get_loss(args):
    y_pred, y_true = args
    loss = y_true - y_pred
    loss = K.mean(loss)
    return loss
        
  label_tensor = Input(shape=(32,))
  loss_tensor = Lambda(get_loss)([output_tensor, label_tensor])

  optimizer = Adam()

  model = Model([input_tensor, label_tensor], [output_tensor])
  mg_model = multi_gpu_model(model, gpus=num_gpu)
  mg_model.add_loss(loss_tensor)

mg_model.compile(optimizer=optimizer)
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
  mg_model.fit([x, y], None, epochs=100, batch_size=100)

