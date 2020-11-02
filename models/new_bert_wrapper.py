import sys
sys.path.append('.')

from NLP_LIB.models.bert_wrapper import BERTWrapper

import random, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

if len(sys.argv) > 1 and sys.argv[1] == 'unittest':

  print('=== UNIT TESTING ===')

  from NLP_LIB.datasets.array_dataset_wrapper import ArrayDatasetWrapper
  data = ArrayDatasetWrapper({
    'values': [
      ['Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World'], # X
      ['Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World'], # Y
      ['Hella', 'Warld','aello', 'World','Hello', 'Uorld','Hello', 'WWrld','HellZ', 'World'], # X Valid
      ['Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World'], # Y Valid
    ]
  })

  from NLP_LIB.transforms.bert_sentencepiece_pretrain_wrapper import BERTSentencePiecePretrainWrapper
  itokens = BERTSentencePiecePretrainWrapper({'column_id': 0, "max_seq_length": 16, "is_input": True, "is_pretrain": True}, data)
  otokens = BERTSentencePiecePretrainWrapper({'column_id': 0, "max_seq_length": 16, "is_input": False, "is_pretrain": True}, data)

  config = {
    'len_limit': 16,
    'd_model': 64,
    'd_inner_hid': 10,
    'n_head': 8,
    'd_k': 64,
    'd_v': 64,
    'layers': 2,
    'dropout': 0.1,
    'share_word_emb': True,
    'max_input_length': 16,
    'max_mask_tokens': 2,
    'cached_data_dir': '_cache_',
  }

  transformer = BERTWrapper(config, itokens, otokens)

  [input_tensors, output_tensors] = transformer.get_forward_tensors()
  print("=== INPUT_TENSOR ===")
  print(input_tensors)
  print("=== OUTPUT_TENSOR ===")
  print(output_tensors)

  train_model = Model(input_tensors, output_tensors)

  metric_funcs = transformer.get_metric_functions()

  from NLP_LIB.ext.bert.optimization import AdamWeightDecayOptimizer

  adamm = AdamWeightDecayOptimizer(learning_rate=0.001,
    num_train_steps=100,
    warmup_steps=10,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    initial_step=0
  )

  train_model.compile(optimizer=adamm, 
    loss=transformer.get_loss_function(),
    metrics=transformer.get_metric_functions()
  )

  print(tf.trainable_variables())
  print(adamm.variables())
  print(len(tf.trainable_variables()))
  print(len(adamm.variables()))

  print(adamm.get_slot_names())
  print(len(adamm.get_slot_names()))  

  train_model.summary()

  print('Start unit testing : New BERTWrapper')

  sess = K.get_session()
  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess.run(init)

  test_data = [
    ['Hello', 'World'],
    ['Hello', 'World'],
    ['Hello', 'World'],
    ['Hello', 'World'],
  ]
  input_vals = itokens.encode(test_data, max_length=16)
  output_vals = otokens.encode(test_data, max_length=16)
  print(input_vals)
  print(output_vals)

  train_model.fit(x=input_vals, y=output_vals, batch_size=1, epochs=1,
    callbacks=[]
  )

  print("=== OUTPUT_TENSOR ===")
  print(output_tensors)

  # After finished training, construct inferencing model!
  immediate_tensors = transformer.get_immediate_tensors()
  print("=== IMMEDIATE_TENSORS ===")
  print(immediate_tensors)

  combined_output_tensors = []
  combined_output_tensors.append(output_tensors)
  combined_output_tensors.append(immediate_tensors['attn_maps'])
  combined_output_tensors.append(immediate_tensors['attn_output_maps'])

  inference_model = Model(input_tensors, combined_output_tensors)
  (y, attn_maps, attn_output_maps) = inference_model.predict(input_vals, batch_size=2)

  print('=== Inputs ===')
  print(input_vals)

  print('=== Label ===')
  print(output_vals)

  print('=== Prediction ===')
  print(y)

  print('=== Attention Map Shape ===')
  print(attn_maps.shape)

  print('=== Attention Output Shape ===')
  print(attn_output_maps.shape)

  print('Finished.')
