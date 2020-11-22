# This unit test is for testing Disagreement Loss between 2 tensors.
# The loss should be in range of [0.0, 1.0]
import tensorflow as tf
import numpy as np

a = [
  [ # Pos 1
    [-1.0, -2.0], # Head 1
    [-1.0, -2.0], # Head 2
  ],
  [ # Pos 2
    [1.0, 2.0], # Head 1
    [1.0, 2.0], # Head 2
  ],
  [ # Pos 3
    [1.0, 2.0], # Head 1
    [1.0, 2.0], # Head 2
  ],
]

b = [
  [ # Pos 1
    [-1.0, -2.0], # Head 1
    [-2.0, -1.0], # Head 2
  ],
  [ # Pos 2
    [1.0, 2.0], # Head 1
    [2.0, 1.0], # Head 2
  ],
  [ # Pos 3
    [1.0, 2.0], # Head 1
    [2.0, 1.0], # Head 2
  ],
]

c = [
  [ # Pos 1
    [0.001, 0.002], # Head 1
    [0.001, 0.002], # Head 2
  ],
  [ # Pos 2
    [-200.0, -100.0], # Head 1
    [0.002, 0.001], # Head 2
  ],
  [ # Pos 3
    [0.001, 0.002], # Head 1
    [0.001, 0.002], # Head 2
  ],
]

def disagree_val(w):
  w_norm = tf.nn.l2_normalize(w, axis=-1)
  w1 = tf.expand_dims(w_norm, 1)
  w2 = tf.expand_dims(w_norm, 2)
  w_mul = tf.multiply(w1, w2)
  c_dist = tf.reduce_sum(w_mul, axis=[-1])
  loss = tf.reduce_mean(c_dist, axis=[-2, -1])

  return ['w_norm', 'w1', 'w2', 'w_mul', 'c_dist', 'loss'], [w_norm, w1, w2, w_mul, c_dist, loss]


ta = tf.constant(a)
tb = tf.constant(b)
tc = tf.constant(c)

with tf.Session() as sess:
  for t in [ta, tb, tc]:
    print('--------------------------------------')
    print('t = ' + str(sess.run(t)))
    
    names, tensors = disagree_val(t)
    vals = sess.run(tensors)
    results = {}
    for name, val in zip(names, vals):
      results[name] = val
    
    w_norm = results['w_norm']
    print(' w_norm = ' + str(w_norm))
    print(' w1 = ' + str(results['w1']))
    print(' w2 = ' + str(results['w2']))
    print(' w1.shape = ' + str(results['w1'].shape))
    print(' w2.shape = ' + str(results['w2'].shape))
    print(' w_mul = ' + str(results['w_mul']))
    print(' w_mul.shape = ' + str(results['w_mul'].shape))
    print(' c_dist = ' + str(results['c_dist']))
    print(' c_dist.shape = ' + str(results['c_dist'].shape))
    print(' loss = ' + str(results['loss']))
    print(' loss.shape = ' + str(results['loss'].shape))

