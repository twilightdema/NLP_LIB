from NLP_LIB.nlp_core.model_wrapper import EncoderModelWrapper, TrainableModelWrapper, SequenceModelWrapper
from NLP_LIB.ext.bert.modeling import BertConfig, BertModel
import random, os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.initializers import *
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers

class BERTWrapper(EncoderModelWrapper, TrainableModelWrapper, SequenceModelWrapper):

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
        'name': 'max_mask_tokens',
        'type': 'int',
        'default': 30,
        'required': True,
        'remark': 'Maximum number of tokens to be masked, in case of training in Masked Language Model objective function.'
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
    return str(self.config['max_input_length']) + super(BERTWrapper, self).get_data_effected_configs() 

  # When initialize model, we check if all required configuration values
  # are available and issue error if not.
  def __init__(self, config, input_data_transform, output_data_transform):    
    super(BERTWrapper, self).__init__(config, input_data_transform, output_data_transform)

    # Inputs to BERT model
    self.input_ids = None
    self.input_mask = None
    self.token_type_ids = None

    # More optional inputs for MLM objective function
    # In inferencing steps, this can be just empty tensor
    self.masked_lm_positions = None,

    self.output_tensor = None
    self.all_encoder_output_tensors = None
    self.encoder_output_tensor = None
    self.mlm_output_tensor = None

    self.loss_tensor = None
    self.accuracy_tensor = None
    self.perplexity_tensor = None
    self.encoder_self_attention_tensor = None
    self.prev_output_tensor = None

  # Function to get Keras input tensors
  def get_input_tensors(self):
    if self.input_ids is None:
      self.input_ids = Input(shape=(self.config['max_input_length'],), dtype='int32')
    if self.input_mask is None:
      self.input_mask = Input(shape=(self.config['max_input_length'],), dtype='int32')
    if self.token_type_ids is None:
      self.token_type_ids = Input(shape=(self.config['max_input_length'],), dtype='int32')
    if self.masked_lm_positions is None:
      self.masked_lm_positions = Input(shape=(self.config['max_mask_tokens'],), dtype='int32')
    return [self.input_ids, self.input_mask, self.token_type_ids, self.masked_lm_positions]

  # Function to get Keras output tensors
  def get_output_tensors(self): 
    if self.output_tensor is None:
      encoder_output_tensor = self.get_encoder_output_tensors()
      self.output_tensor = encoder_output_tensor # self.transformer.target_layer(encoder_output_tensor)
    return self.output_tensor

  # Function to get other tensors those are specific to each model. Result map from name to tensor object.
  def get_immediate_tensors(self):
    return {
      'all_encoder_output_tensors': self.all_encoder_output_tensors,
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

      def model_fn(all_inputs):

        input_ids, input_mask, token_type_ids = all_inputs

        bert_config = BertConfig(
          vocab_size=64,
          hidden_size=64,
          num_hidden_layers=2,
          num_attention_heads=2,
          intermediate_size=10,
          hidden_act='gelu',
          hidden_dropout_prob=0.1,
          attention_probs_dropout_prob=0.1,
          max_position_embeddings=256,
          type_vocab_size=2,
          initializer_range=0.02
        )
        bert = BertModel(
          bert_config=bert_config,
          is_training=True,
          input_ids=input_ids,
          input_mask=input_mask,
          token_type_ids=token_type_ids,
          use_one_hot_embeddings=True,
          scope=None,
          embedding_size=64,
          input_embeddings=None,
          input_reprs=None,
          update_embeddings=True,
          untied_embeddings=False
        )

        return bert.get_all_encoder_layers()

      
        '''
        self.transformer = Transformer(self.input_data_transform, self.output_data_transform, 
          self.config['len_limit'], self.config['d_model'], 
          self.config['d_inner_hid'], self.config['n_head'],
          self.config['d_k'], self.config['d_v'],
          self.config['layers'], self.config['dropout'],
          self.config['share_word_emb'],
          self.config['share_transformer_weights'],
        )
        '''

      # Encoder Side
      input_ids, input_mask, token_type_ids = self.get_input_tensors()
      all_encoder_output_tensors = Lambda(model_fn, name='bert_encoder')([input_ids, input_mask, token_type_ids])

      self.all_encoder_output_tensors = all_encoder_output_tensors
      self.encoder_output_tensor = all_encoder_output_tensors[-1]
      #src_pos = Lambda(self.transformer.get_pos_seq)(input_tensor)
      #self.encoder_output_tensor, self.encoder_self_attention_tensor = self.transformer.encoder(input_tensor, src_pos, return_att=True, active_layers=999)
    return self.encoder_output_tensor

  # In BERT, we separate MLM outputs from encoder outputs.
  # Basically, MLM outputs are encoder outputs passing Dense and Softmax to get prediction tokens.
  def get_mlm_output_tensors(self):
    if self.mlm_output_tensor is None:
      [_,_,_,masked_lm_positions] = self.get_input_tensors()
      encoder_output_tensor = self.get_encoder_output_tensors()
    
      def mlm_prediction_fn(all_inputs):

        encoder_sequence_output, masked_lm_positions = all_inputs

        """Masked language modeling softmax layer."""
        with tf.variable_scope("mlm_predictions"):
          relevant_hidden = pretrain_helpers.gather_positions(
              encoder_sequence_output, masked_lm_positions)
          hidden = tf.layers.dense(
              relevant_hidden,
              units=modeling.get_shape_list(model.get_embedding_table())[-1],
              activation=modeling.get_activation(self._bert_config.hidden_act),
              kernel_initializer=modeling.create_initializer(
                  self._bert_config.initializer_range))
          hidden = modeling.layer_norm(hidden)
          output_bias = tf.get_variable(
              "output_bias",
              shape=[self._bert_config.vocab_size],
              initializer=tf.zeros_initializer())
          logits = tf.matmul(hidden, model.get_embedding_table(),
                            transpose_b=True)
          logits = tf.nn.bias_add(logits, output_bias)

          probs = tf.nn.softmax(logits)
          log_probs = tf.nn.log_softmax(logits)
          preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

          return [logits, probs, log_probs, preds]

      self.mlm_output_tensor = Lambda(mlm_prediction_fn, name='mlm_prediction')([encoder_output_tensor, masked_lm_positions])

    return self.mlm_output_tensor

  #############################################################
  ## Subclass for ModelWrapper  
  # Function to contruct Keras tensors for running model in forward loop.
  # Return list of [inputs, outputs] tensors.
  def get_forward_tensors(self):

    # Predict prev_output shifted left plus additional new token from Decoder Output
    input_tensor = self.get_input_tensors()
    prev_output_tensor = self.get_prev_output_tensors()
    output_tensor = self.get_output_tensors()

    return [[*input_tensor, prev_output_tensor], [output_tensor]]

  #############################################################
  ## Subclass for TrainableModelWrapper  
  # Function to encode input based on model configuration
  def encode_output(self, output_tokens):
    return self.output_data_transform.encode(output_tokens, self.config['max_input_length'])

  def _get_masked_lm_output(self, 
    masked_lm_weights, 
    masked_lm_ids,
    masked_lm_positions,
    encoder_sequence_output,
    embedding_table
  ):

    def loss_fn(all_inputs):

      masked_lm_weights, masked_lm_ids, masked_lm_positions, encoder_sequence_output = all_inputs

      """Masked language modeling softmax layer."""
      with tf.variable_scope("generator_predictions"):
        relevant_hidden = pretrain_helpers.gather_positions(
            encoder_sequence_output, masked_lm_positions)
        hidden = tf.layers.dense(
            relevant_hidden,
            units=modeling.get_shape_list(model.get_embedding_table())[-1],
            activation=modeling.get_activation(self._bert_config.hidden_act),
            kernel_initializer=modeling.create_initializer(
                self._bert_config.initializer_range))
        hidden = modeling.layer_norm(hidden)
        output_bias = tf.get_variable(
            "output_bias",
            shape=[self._bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(hidden, model.get_embedding_table(),
                          transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)

        oh_labels = tf.one_hot(
            inputs.masked_lm_ids, depth=self._bert_config.vocab_size,
            dtype=tf.float32)

        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)
        label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)

        numerator = tf.reduce_sum(masked_lm_weights * label_log_probs)
        denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
        loss = numerator / denominator
        preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

        return MLMOutput(
            logits=logits, probs=probs, per_example_loss=label_log_probs,
            loss=loss, preds=preds)

  # Function to return loss function for the model
  def get_loss_function(self):
    if self.loss_tensor is None:
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
    return self.loss_tensor    

  # Function to get list of metric name for this model when perform training.
  def get_metric_names(self):
    return ['acc', 'ppl']

  # Function to get list of metric function for this model when perform training.
  def get_metric_functions(self):
    if self.accuracy_tensor is None:
      input_tensors = self.get_input_tensors()
      def acc(y_true, y_pred):
        # Remove dummy Dense dimension to get sparse dimension
        y_true = Lambda(lambda x:x[:,:,0])(y_true)
        mask2 = tf.cast(tf.equal(input_tensors, 4), 'float32') # Mask non <MASK>
        corr = K.cast(K.equal(K.cast(y_true, 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
        corr = K.sum(corr * mask2, -1) / ( K.sum(mask2, -1) + 1e-10 )
        return K.mean(corr)
      self.accuracy_tensor = acc

    if self.perplexity_tensor is None:
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

    return [self.accuracy_tensor, self.perplexity_tensor]
   
# Unit Test
print('-===================-')
print(__name__)
#if __name__ == '__main__':

if __name__ == 'tensorflow.keras.initializers':

  print('=== UNIT TESTING ===')

  from NLP_LIB.datasets.array_dataset_wrapper import ArrayDatasetWrapper
  data = ArrayDatasetWrapper({
    'values': [
      ['Hello', 'World'], # X
      ['Hello', 'World'], # Y
      ['Hello', 'World'], # X Valid
      ['Hello', 'World'], # Y Valid
    ]
  })

  from NLP_LIB.transforms.fullword_dictionary_wrapper import FullWordDictionaryWrapper
  itokens = FullWordDictionaryWrapper({'column_id': 0}, data)
  otokens = FullWordDictionaryWrapper({'column_id': 1}, data)

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
    'cached_data_dir': '_cache_',
  }

  transformer = BERTWrapper(config, itokens, otokens)
  [input_tensors, output_tensors] = transformer.get_forward_tensors()
  print("=== INPUT_TENSOR ===")
  print(input_tensors)
  print("=== OUTPUT_TENSOR ===")
  print(output_tensors)
  model = Model(input_tensors, output_tensors)

  metric_funcs = transformer.get_metric_functions()

  model.compile(optimizer='adam', 
    loss=transformer.get_loss_function(),
    metrics=metric_funcs
  )

  model.summary()

  exit(0)

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
  exit(0)

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