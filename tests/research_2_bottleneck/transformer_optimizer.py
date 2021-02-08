import tensorflow as tf
import numpy as np
from official import nlp
import official.nlp.optimization

class GPTLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, 
               d_model, 
               warmup_steps=4000):
    super(GPTLearningRateSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  def get_config(self):
    return {
        'd_model': self.initial_learning_rate,
        'warmup_steps': self.warmup_steps
    }

class BERTLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule."""

  def __init__(self,
               initial_learning_rate,
               num_train_steps,
               warmup_steps,
               power=1.0,
               name=None):
    super(BERTLearningRateSchedule, self).__init__()

    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=initial_learning_rate,
      decay_steps=num_train_steps - warmup_steps,
      end_learning_rate=0)

    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = lr_schedule
    self.name = name
    self.num_train_steps = num_train_steps

  def __call__(self, step):
    with tf.name_scope(self.name or 'WarmUp') as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = (
          self.initial_learning_rate *
          tf.math.pow(warmup_percent_done, self.power))
      output = tf.where(
        global_step_float < warmup_steps_float, 
        warmup_learning_rate,
        self.decay_schedule_fn(step - self.warmup_steps),
        name=name
        )
      return output
      '''
      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: self.decay_schedule_fn(step),
          name=name)
      '''

  def get_config(self):
    return {
        'initial_learning_rate': self.initial_learning_rate,
        'decay_schedule_fn': self.decay_schedule_fn,
        'warmup_steps': self.warmup_steps,
        'power': self.power,
        'name': self.name
    }

def create_bert_optimizer(
               initial_learning_rate,
               num_train_steps,
               warmup_steps,
               power=1.0,
               weight_decay_rate=0.01,
               epsilon=1e-6,
               name=None):
  bert_lr = BERTLearningRateSchedule(initial_learning_rate, num_train_steps, warmup_steps, power, name)
  optimizer = nlp.optimization.AdamWeightDecay(
        learning_rate=bert_lr,
        weight_decay_rate=weight_decay_rate,
        epsilon=epsilon,
        exclude_from_weight_decay=['LayerNorm', 'layer_norm', 'bias'])
  return optimizer
