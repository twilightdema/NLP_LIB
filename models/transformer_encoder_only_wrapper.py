from NLP_LIB.nlp_core.model_wrapper import EncoderModelWrapper, TrainableModelWrapper, SequenceModelWrapper
from NLP_LIB.ext.transformer import Transformer
import random, os, sys
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

import tensorflow as tf

class TransformerEncoderOnlyWrapper(EncoderModelWrapper, TrainableModelWrapper, SequenceModelWrapper):

  def get_configuration_list(self):
    conf_list = super().get_configuration_list()
    conf_list.extend([
      {
        'name': 'len_limit',
        'type': 'int',
        'default': 64,
        'required': True,
        'remark': 'Max length of input sequence'
      },
      {
        'name': 'd_model',
        'type': 'int',
        'default': 256,
        'required': True,
        'remark': 'Document embedding dimension'
      },
      {
        'name': 'd_inner_hid',
        'type': 'int',
        'default': 512,
        'required': True,
        'remark': 'Model hidden dimension'
      },
      {
        'name': 'n_head',
        'type': 'int',
        'default': 4,
        'required': True,
        'remark': 'Number of attention head'
      },
      {
        'name': 'd_k',
        'type': 'int',
        'default': 64,
        'required': True,
        'remark': 'Attention Key size'
      },
      {
        'name': 'd_v',
        'type': 'int',
        'default': 64,
        'required': True,
        'remark': 'Attention Value size'
      },
      {
        'name': 'layers',
        'type': 'int',
        'default': 2,
        'required': True,
        'remark': 'Transformer layer count'
      },
      {
        'name': 'dropout',
        'type': 'float',
        'default': 0.1,
        'required': True,
        'remark': 'Dropout percentage used in training model'
      },
      {
        'name': 'share_word_emb',
        'type': 'bool',
        'default': True,
        'required': True,
        'remark': 'Whether input/output share same word embedding or not'
      },
      {
        'name': 'max_input_length',
        'type': 'int',
        'default': 256,
        'required': True,
        'remark': 'Maximum input length, Can be None if we are not running deterministic shape. Finetune on classification task normally need this value explicitly defined.'
      },
      {
        'name': 'train_mask_only',
        'type': 'bool',
        'default': False,
        'required': True,
        'remark': 'Flag indicate whether we train only <MASK> input position or not.'
      },
      {
        'name': 'share_transformer_weights',
        'type': 'bool',
        'default': False,
        'required': False,
        'remark': 'If set to True, all transformer layers have their weights shared. This technique was proposed in ALBERT paper.'
      },
    ])
    return conf_list

  # Function to return list of objects to differentiate cached of input/output that model will use.
  # Basically it is configurations that effect encoded data.
  def get_data_effected_configs(self):
    # For transformer, max_input_lenth effect length of encoded input/output
    return str(self.config['max_input_length']) + super(TransformerEncoderOnlyWrapper, self).get_data_effected_configs() 

  # When initialize model, we check if all required configuration values
  # are available and issue error if not.
  def __init__(self, config, input_data_transform, output_data_transform):    
    super(TransformerEncoderOnlyWrapper, self).__init__(config, input_data_transform, output_data_transform)
    self.input_tensor = None
    self.output_tensor = None
    self.encoder_output_tensor = None
    self.loss_tensor = None
    self.accuracy_tensor = None
    self.perplexity_tensor = None
    self.encoder_self_attention_tensor = None
    self.prev_output_tensor = None

  # Function to get Keras input tensors
  def get_input_tensors(self):
    if self.input_tensor is None:
      self.input_tensor = Input(shape=(self.config['max_input_length'],), dtype='int32')
    return self.input_tensor

  # Function to get Keras output tensors
  def get_output_tensors(self):
    if self.output_tensor is None:
      encoder_output_tensor = self.get_encoder_output_tensors()
      self.output_tensor = self.transformer.target_layer(encoder_output_tensor)
    return self.output_tensor

  # Function to get other tensors those are specific to each model. Result map from name to tensor object.
  def get_immediate_tensors(self):
    return {
      'encoder_self_attention': self.encoder_self_attention_tensor,
    }

  # Function to encode input based on model configuration
  def encode_input(self, input_tokens):
    return self.input_data_transform.encode(input_tokens, self.config['max_input_length'])

  #############################################################
  ## Subclass for SequenceModelWrapper  
  # Function to get Keras previous output tensors (for Sequence Model)
  def get_prev_output_tensors(self):
    if self.prev_output_tensor is None:
      self.prev_output_tensor = Input(shape=(self.config['max_input_length'],), dtype='int32')
    return self.prev_output_tensor

  #############################################################
  ## Subclass for EncoderModelWrapper  
  # Function to return Keras Tensors of encoder output.
  def get_encoder_output_tensors(self):
    if self.encoder_output_tensor is None:
      self.transformer = Transformer(self.input_data_transform, self.output_data_transform, 
        self.config['len_limit'], self.config['d_model'], 
        self.config['d_inner_hid'], self.config['n_head'],
        self.config['d_k'], self.config['d_v'],
        self.config['layers'], self.config['dropout'],
        self.config['share_word_emb'],
        self.config['share_transformer_weights'],
      )
      # Encoder Side
      input_tensor = self.get_input_tensors()
      src_pos = Lambda(self.transformer.get_pos_seq)(input_tensor)
      self.encoder_output_tensor, self.encoder_self_attention_tensor = self.transformer.encoder(input_tensor, src_pos, return_att=True, active_layers=999)
    return self.encoder_output_tensor

  #############################################################
  ## Subclass for ModelWrapper  
  # Function to contruct Keras tensors for running model in forward loop.
  # Return list of [inputs, outputs] tensors.
  def get_forward_tensors(self):

    # Predict prev_output shifted left plus additional new token from Decoder Output
    input_tensor = self.get_input_tensors()
    prev_output_tensor = self.get_prev_output_tensors()
    output_tensor = self.get_output_tensors()

    return [[input_tensor, prev_output_tensor], [output_tensor]]

  #############################################################
  ## Subclass for TrainableModelWrapper  
  # Function to encode input based on model configuration
  def encode_output(self, output_tokens):
    return self.output_data_transform.encode(output_tokens, self.config['max_input_length'])

  # Function to return loss function for the model
  def get_loss_function(self):
    if self.loss_tensor is None:
      if self.config['train_mask_only'] == True:
        input_tensors = self.get_input_tensors()
        def get_loss(y_true, y_pred):
          # Remove dummy Dense dimension to get sparse dimension
          y_true = Lambda(lambda x:x[:,:,0])(y_true)
          y_true = tf.cast(y_true, 'int32')

          #y_true = tf.Print(y_true, ['y_true', tf.shape(y_true), y_true])
          #y_pred = tf.Print(y_pred, ['y_pred', tf.shape(y_pred), y_pred])

          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
          mask2 = tf.cast(tf.equal(input_tensors, 4), 'float32') # Mask non <MASK>

          #loss = tf.Print(loss, ['loss_', tf.shape(loss), loss], summarize=32)
          #mask2 = tf.Print(mask2, ['mask2', tf.shape(mask2), mask2])

          denom = tf.reduce_sum(mask2, -1)
          #denom = tf.Print(denom, ['denom', tf.shape(denom), denom], summarize=32)

          nom = tf.reduce_sum(loss * mask2, -1)
          #nom = tf.Print(nom, ['nom', tf.shape(nom), nom], summarize=32)

          #loss_zero = tf.zeros(tf.shape(nom), dtype=nom.dtype)          
          #loss = tf.where(tf.equal(denom, 0), loss_zero, nom / denom)
          loss = nom / (denom + 1e-10)

          #loss = tf.Print(loss, ['loss', tf.shape(loss), loss], summarize=32)
          loss = K.mean(loss)
          #loss = tf.Print(loss, ['meanloss', tf.shape(loss), loss], summarize=32)
          return loss
        self.loss_tensor = get_loss
      else:
        def get_loss(y_true, y_pred):
          # Remove dummy Dense dimension to get sparse dimension
          y_true = Lambda(lambda x:x[:,:,0])(y_true)
          y_true = tf.cast(y_true, 'int32')
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
          mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
          loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
          loss = K.mean(loss)
          return loss
        self.loss_tensor = get_loss
    return self.loss_tensor    

  # Function to get list of metric name for this model when perform training.
  def get_metric_names(self):
    return ['acc', 'ppl']

  # Function to get list of metric function for this model when perform training.
  def get_metric_functions(self):
    if self.accuracy_tensor is None:
      if self.config['train_mask_only'] == True:
        input_tensors = self.get_input_tensors()
        def acc(y_true, y_pred):
          # Remove dummy Dense dimension to get sparse dimension
          y_true = Lambda(lambda x:x[:,:,0])(y_true)
          mask2 = tf.cast(tf.equal(input_tensors, 4), 'float32') # Mask non <MASK>
          corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
          corr = K.sum(corr * mask2, -1) / ( K.sum(mask2, -1) + 1e-10 )
          return K.mean(corr)
        self.accuracy_tensor = acc
      else:
        def acc(y_true, y_pred):
          # Remove dummy Dense dimension to get sparse dimension
          y_true = Lambda(lambda x:x[:,:,0])(y_true)
          mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
          corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
          corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
          return K.mean(corr)
        self.accuracy_tensor = acc

    if self.perplexity_tensor is None:
      if self.config['train_mask_only'] == True:
        input_tensors = self.get_input_tensors()
        def ppl(y_true, y_pred):
          # Remove dummy Dense dimension to get sparse dimension
          y_true = Lambda(lambda x:x[:,:,0])(y_true)
          y_true = tf.cast(y_true, 'int32')
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
          mask2 = tf.cast(tf.equal(input_tensors, 4), 'float32') # Mask non <MASK>
          loss = tf.reduce_sum(loss * mask2, -1) / ( tf.reduce_sum(mask2, -1) + 1e-10 )
          loss = K.mean(loss)
          ppl = K.exp(loss)
          return ppl
        self.perplexity_tensor = ppl
      else:
        def ppl(y_true, y_pred):
          # Remove dummy Dense dimension to get sparse dimension
          y_true = Lambda(lambda x:x[:,:,0])(y_true)
          y_true = tf.cast(y_true, 'int32')
          loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
          mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
          loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
          loss = K.mean(loss)
          ppl = K.exp(loss)
          return ppl
        self.perplexity_tensor = ppl

    return [self.accuracy_tensor, self.perplexity_tensor]


  ''' TO BE DELTETED
  # Function to return Keras Tensors of loss of the model.
  def get_loss_tensors(self):
    if self.loss_tensor is None:

      def get_loss(args):
        y_pred, y_true = args
        y_true = tf.cast(y_true, 'int32')
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
        loss = tf.reduce_sum(loss * mask, -1) / tf.reduce_sum(mask, -1)
        loss = K.mean(loss)
        return loss

      def get_accu(args):
        y_pred, y_true = args
        mask = tf.cast(tf.not_equal(y_true, 0), 'float32')
        corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
        corr = K.sum(corr * mask, -1) / K.sum(mask, -1)
        return K.mean(corr)
      
      output_tensor = self.get_output_tensors()
      label_tensor = self.get_label_tensors()
      tgt_true = Lambda(lambda x:x[:,1:])(label_tensor)    

      self.loss_tensor = Lambda(get_loss)([output_tensor, tgt_true])

      # Additional Metrics for Transformer
      self.perplexity_tensor = Lambda(K.exp)(self.loss_tensor)
      self.accuracy_tensor = Lambda(get_accu)([output_tensor, tgt_true])

    return self.loss_tensor

  # Function to return Keras Tensors of label of the model.
  def get_label_tensors(self):
    if self.label_tensor is None:
      self.label_tensor = Input(shape=(self.config['max_input_length'],), dtype='int32')
    return self.label_tensor

  # Function to contruct Keras tensors for running model backpropagation loop.
  # Return list of [labels, losses] tensors.
  def get_gradient_tensors(self):
    label_tensor = self.get_label_tensors()
    loss_tensor = self.get_loss_tensors()
    return [[label_tensor], [loss_tensor]]

  # Function to get list of metric name for this model when perform training.
  def get_metric_names(self):
    return ['acc', 'ppl']

  # Function to get list of metric tensor for this model when perform training.
  def get_metric_tensors(self):
    return [self.accuracy_tensor, self.perplexity_tensor]
  '''

'''    
# Unit Test
if __name__ == '__main__':

  from transforms.fullword_dictionary_wrapper import FullWordDictionaryWrapper

  itokens = FullWordDictionaryWrapper(list('0123456789'))
  otokens = FullWordDictionaryWrapper(list('0123456789'))

  config = {
    'len_limit': 64,
    'd_model': 64,
    'd_inner_hid': 64,
    'n_head': 4,
    'd_k': 64,
    'd_v': 64,
    'layers': 2,
    'dropout': 0.1,
    'share_word_emb': True,
    'max_input_length': 256,
  }
  transformer = TransformerWrapper(config, itokens, otokens)
  [input_tensors, output_tensors] = transformer.get_forward_tensors()
  [label_tensors, loss_tensors] = transformer.get_gradient_tensors()

  print('Start unit testing')

  def GenSample():
    x = random.randint(0, 99999)
    y = hex(x);  x = str(x)
    return x, y

  X, Y = [], []
  for _ in range(100):
    x, y = GenSample()
    X.append(list(x))
    Y.append(list(y))

  X = transformer.encode_input(X)
  Y = transformer.encode_output(Y)
  print(X.shape, Y.shape)

  model = Model([*input_tensors, *label_tensors], [*output_tensors, *loss_tensors])
  model.summary()
  model.add_loss(loss_tensors[0])  
  model.compile('adam')
  model.summary()
  model.fit(x=[X, Y, Y], y=None, batch_size=5, epochs=10,
    validation_split=0.05,
    callbacks=[]
  )

  print('Finished.')
'''