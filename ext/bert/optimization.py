# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions and classes related to optimization (weight updates).
Modified from the original BERT code to allow for having separate learning
rates for different layers of the network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

import traceback

def create_optimizer(
    loss, learning_rate, num_train_steps, weight_decay_rate=0.0, use_tpu=False,
    warmup_steps=0, warmup_proportion=0, lr_decay_power=1.0,
    layerwise_lr_decay_power=-1, n_transformer_layers=None):
  """Creates an optimizer and training op."""
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=lr_decay_power,
      cycle=False)
  warmup_steps = max(num_train_steps * warmup_proportion, warmup_steps)
  learning_rate *= tf.minimum(
      1.0, tf.cast(global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))

  if layerwise_lr_decay_power > 0:
    learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power,
                                   n_transformer_layers)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=weight_decay_rate,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  if use_tpu:
    optimizer = tf.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op


class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               initial_step,
               learning_rate,

               num_train_steps,
               warmup_steps=0, 
               warmup_proportion=0, 
               lr_decay_power=1.0,

               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.initial_step = initial_step
    self.learning_rate = learning_rate
    self.num_train_steps = num_train_steps
    self.warmup_steps = warmup_steps
    self.warmup_proportion = warmup_proportion
    self.lr_decay_power = lr_decay_power
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay
    self.learning_rate_tensor = tf.Variable(0.0, shape=[], dtype=tf.float32, name='learning_rate')
    # self.mv_lookup = {}

  def _create_slots(self, var_list):
    print('>>>>>>>>>>> _create_slots is called.')
    for param in var_list:
      if param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = self._zeros_slot(param, param_name + "/adam_m", param_name + "/adam_m")
      v = self._zeros_slot(param, param_name + "/adam_v", param_name + "/adam_v")

      '''
      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      '''

      # self.mv_lookup[param_name] = (m, v)

  def _apply_gradients(self, grads_and_vars, learning_rate):
    print('_apply_gradients is called!!!')
    """See base class."""

    # Create slot variables
    var_list = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue
      var_list.append(param)
    with ops.init_scope():
      self._create_slots(var_list)

    # Build training operations
    assignments = []
    check_values = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      #m, v = self.mv_lookup[param_name]
      m = self.get_slot(param, param_name + "/adam_m")
      v = self.get_slot(param, param_name + "/adam_v")

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))
      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      check_update_nan = tf.Assert(tf.logical_not(tf.reduce_all(tf.is_nan(update))), [param_name, 'NAN update', update])
      check_update_inf = tf.Assert(tf.logical_not(tf.reduce_all(tf.is_inf(update))), [param_name, 'INF update', update])
      check_values.append(check_update_nan)
      check_values.append(check_update_inf)
      #update = 0

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self.weight_decay_rate > 0:
        if self._do_use_weight_decay(param_name):
          update += self.weight_decay_rate * param

      update_with_lr = learning_rate * update
      # update_with_lr = tf.Print(update_with_lr, ['\nupdate_with_lr', param_name, tf.shape(update_with_lr), update_with_lr], summarize=32)
      max_update_with_lr = tf.reduce_max(update_with_lr)
      min_update_with_lr = tf.reduce_min(update_with_lr)
      # update_with_lr = tf.Print(update_with_lr, ['\nupdate_with_lr', param_name, tf.shape(update_with_lr), min_update_with_lr, max_update_with_lr], summarize=32)

      check_update_with_lr_nan = tf.Assert(tf.logical_not(tf.reduce_all(tf.is_nan(update_with_lr))), [param_name, 'NAN update_with_lr', update_with_lr])
      check_update_with_lr_inf = tf.Assert(tf.logical_not(tf.reduce_all(tf.is_inf(update_with_lr))), [param_name, 'INF update_with_lr', update_with_lr])
      check_values.append(check_update_with_lr_nan)
      check_values.append(check_update_with_lr_inf)

      next_param = param - update_with_lr

      check_next_param_nan = tf.Assert(tf.logical_not(tf.reduce_all(tf.is_nan(next_param))), [param_name, 'NAN next_param', next_param])
      check_next_param_inf = tf.Assert(tf.logical_not(tf.reduce_all(tf.is_inf(next_param))), [param_name, 'INF next_param', next_param])
      check_values.append(check_next_param_nan)
      check_values.append(check_next_param_inf)

      # Ensure that the debug operations are executed.
      for op in check_values:
        op.mark_used()

      '''
      assignments.extend(
          [param.assign(next_param),]
      )
      '''
      assignments.extend(
          [
           param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)
          ]
          )
      assignments.extend(check_values)

    return assignments

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):

    # Emulate global step to the optimizer, we will increment it every "apply_gradients" call.
    # and also use it to perform Warmup ramping and Weight decay.
    internal_global_step_name = self._get_variable_name("global_step_tf")
    with tf.init_scope():
      internal_global_step = self._get_or_make_slot(global_step, tf.constant(self.initial_step, dtype=tf.float32), internal_global_step_name, internal_global_step_name)

    print('self.initial_step = ' + str(self.initial_step))

    learning_rate = tf.train.polynomial_decay(
        self.learning_rate,
        internal_global_step,
        self.num_train_steps,
        end_learning_rate=0.0,
        power=self.lr_decay_power,
        cycle=False)

    warmup_steps = max(self.num_train_steps * self.warmup_proportion, self.warmup_steps)

    '''
    if layerwise_lr_decay_power > 0:
      learning_rate = _get_layer_lrs(learning_rate, layerwise_lr_decay_power,
                                    n_transformer_layers)
    '''

    internal_global_step = self.get_slot(global_step, internal_global_step_name)
    # global_step_print = tf.Print(internal_global_step, ['internal_global_step', tf.shape(internal_global_step), internal_global_step], summarize=32)
    global_step_update_op = internal_global_step.assign(internal_global_step + 1)

    print('warmup_steps = ' + str(warmup_steps))
    print('num_train_steps = ' + str(self.num_train_steps))

    learning_rate *= tf.minimum(
        1.0, tf.cast(internal_global_step, tf.float32) / tf.cast(warmup_steps, tf.float32))
    #lr_print = tf.Print(learning_rate, ['learning_rate', tf.shape(learning_rate), learning_rate], summarize=32)

    lr_update_op = self.learning_rate_tensor.assign(learning_rate)

    # Clip the gradient to be at most 1.0 (from original BERT implementation)
    grads, tvars = zip(*grads_and_vars)
    (clipped_grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)
    grads_and_vars = list(zip(clipped_grads, tvars))

    if isinstance(learning_rate, dict):
      key_to_grads_and_vars = {}
      for grad, var in grads_and_vars:
        update_for_var = False
        for key in self.learning_rate:
          if key in var.name:
            update_for_var = True
            if key not in key_to_grads_and_vars:
              key_to_grads_and_vars[key] = []
            key_to_grads_and_vars[key].append((grad, var))
        if not update_for_var:
          raise ValueError("No learning rate specified for variable", var)
      assignments = []
      for key, key_grads_and_vars in key_to_grads_and_vars.items():
        assignments += self._apply_gradients(key_grads_and_vars,
                                             learning_rate[key])
    else:
      assignments = self._apply_gradients(grads_and_vars, learning_rate)
    return tf.group([*assignments, global_step_update_op, lr_update_op], name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name


def _get_layer_lrs(learning_rate, layer_decay, n_layers):
  """Have lower learning rates for layers closer to the input."""
  key_to_depths = collections.OrderedDict({
      "/embeddings/": 0,
      "/embeddings_project/": 0,
      "task_specific/": n_layers + 2,
  })
  for layer in range(n_layers):
    key_to_depths["encoder/layer_" + str(layer) + "/"] = layer + 1
  return {
      key: learning_rate * (layer_decay ** (n_layers + 2 - depth))
      for key, depth in key_to_depths.items()
  }
