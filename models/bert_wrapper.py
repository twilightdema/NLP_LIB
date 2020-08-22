import sys
sys.path.append('.')

from NLP_LIB.nlp_core.model_wrapper import EncoderModelWrapper, TrainableModelWrapper, SequenceModelWrapper
from NLP_LIB.ext.bert.modeling import BertConfig, BertModel, get_shape_list, get_activation, create_initializer, layer_norm
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

    # Pointer to BERT model
    self.bert_config = None
    self.bert = None

    # Inputs to BERT model
    self.input_ids = None
    self.input_mask = None
    self.token_type_ids = None

    # More optional inputs for MLM objective function
    # In inferencing steps, this can be just empty tensor
    self.masked_lm_positions = None
    self.masked_lm_weights = None

    self.output_tensor = None
    self.all_encoder_output_tensors = None
    self.encoder_output_tensor = None
    self.mlm_output_tensor = None

    self.loss_tensor = None

    # Tensorflow immediate tensors for metrics calculation
    self.tf_preds = None
    self.tf_per_example_loss = None

    self.accuracy_tensor = None
    self.perplexity_tensor = None
    self.encoder_self_attention_tensor = None
    self.prev_output_tensor = None

  # Function to get Keras input tensors
  def get_input_tensors(self):
    if self.input_ids is None:
      self.input_ids = Input(name='input_ids', shape=(self.config['max_input_length'],), dtype='int32')
    if self.input_mask is None:
      self.input_mask = Input(name='input_mask', shape=(self.config['max_input_length'],), dtype='int32')
    if self.token_type_ids is None:
      self.token_type_ids = Input(name='token_type_ids', shape=(self.config['max_input_length'],), dtype='int32')
    if self.masked_lm_positions is None:
      self.masked_lm_positions = Input(name='masked_lm_positions', shape=(self.config['max_mask_tokens'],), dtype='int32')
    if self.masked_lm_weights is None:
      self.masked_lm_weights = Input(name='masked_lm_weights', shape=(self.config['max_mask_tokens'],), dtype='float32')
    print('>>>>> self.masked_lm_positions = ' + str(self.masked_lm_positions))
    print('>>>>> self.token_type_ids = ' + str(self.token_type_ids))
    return [self.input_ids, self.input_mask, self.token_type_ids, self.masked_lm_positions, self.masked_lm_weights]

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
      self.prev_output_tensor = Input(name='prev_output', shape=(self.config['max_input_length'],), dtype='int32')
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

        self.bert_config = bert_config
        self.bert = bert

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
      input_ids, input_mask, token_type_ids, _, _ = self.get_input_tensors()

      # TODO: Apparently if we use tf.keras, we do not need to create Lambda layer explicitly anymore!
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
      _,_,_,masked_lm_positions,masked_lm_weights = self.get_input_tensors()
      print('masked_lm_positions = ' + str(masked_lm_positions))
      encoder_output_tensor = self.get_encoder_output_tensors()

      def gather_positions(sequence, positions):
        """Gathers the vectors at the specific positions over a minibatch.

        Args:
          sequence: A [batch_size, seq_length] or
              [batch_size, seq_length, depth] tensor of values
          positions: A [batch_size, n_positions] tensor of indices

        Returns: A [batch_size, n_positions] or
          [batch_size, n_positions, depth] tensor of the values at the indices
        """
        shape = get_shape_list(sequence, expected_rank=[2, 3])
        depth_dimension = (len(shape) == 3)
        if depth_dimension:
          B, L, D = shape
        else:
          B, L = shape
          D = 1
          sequence = tf.expand_dims(sequence, -1)
        position_shift = tf.expand_dims(L * tf.range(B), -1)
        print('positions = ' + str(positions))
        print('positions = ' + str(position_shift))
        flat_positions = tf.reshape(positions + position_shift, [-1])
        flat_sequence = tf.reshape(sequence, [B * L, D])
        gathered = tf.gather(flat_sequence, flat_positions)
        if depth_dimension:
          return tf.reshape(gathered, [B, -1, D])
        else:
          return tf.reshape(gathered, [B, -1])

      def mlm_prediction_fn(all_inputs):

        encoder_sequence_output, masked_lm_positions = all_inputs

        """Masked language modeling softmax layer."""
        with tf.variable_scope("mlm_predictions"):
          relevant_hidden = gather_positions(
              encoder_sequence_output, masked_lm_positions)
          hidden = tf.layers.dense(
              relevant_hidden,
              units=get_shape_list(self.bert.get_embedding_table())[-1],
              activation=get_activation(self.bert_config.hidden_act),
              kernel_initializer=create_initializer(
                  self.bert_config.initializer_range))
          hidden = layer_norm(hidden)
          output_bias = tf.get_variable(
              "output_bias",
              shape=[self.bert_config.vocab_size],
              initializer=tf.zeros_initializer())
          logits = tf.matmul(hidden, self.bert.get_embedding_table(),
                            transpose_b=True)
          logits = tf.nn.bias_add(logits, output_bias)

          probs = tf.nn.softmax(logits)
          log_probs = tf.nn.log_softmax(logits)
          preds = tf.argmax(log_probs, axis=-1, output_type=tf.int32)

          print('[DEBUG] preds = ' + str(preds))
          
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
    mlm_output_tensor = self.get_mlm_output_tensors()
    preds = mlm_output_tensor[3]

    return [[*input_tensor], preds]

  #############################################################
  ## Subclass for TrainableModelWrapper  
  # Function to encode input based on model configuration
  def encode_output(self, output_tokens):
    return self.output_data_transform.encode(output_tokens, self.config['max_input_length'])

  # Function to return loss function for the model
  def get_loss_function(self):
    if self.loss_tensor is None:

      _, _, _, _, masked_lm_weights = self.get_input_tensors()
      _, _, log_probs, _ = self.get_mlm_output_tensors()

      def loss_fn(y_true, y_pred):

        print('[DEBUG] y_true = ' + str(y_true))

        # y_true is IDs of masked tokens
        masked_lm_ids = y_true

        print('[DEBUG] y_pred = ' + str(y_pred))

        print('[DEBUG] log_probs = ' + str(log_probs))

        # y_pred is preds
        preds = y_pred

        """Masked language modeling softmax layer."""
        with tf.variable_scope("mlm_loss"):
          oh_labels = tf.one_hot(
              masked_lm_ids, depth=self.bert_config.vocab_size,
              dtype=tf.float32)

          label_log_probs = -tf.reduce_sum(log_probs * oh_labels, axis=-1)
          numerator = tf.reduce_sum(masked_lm_weights * label_log_probs)
          denominator = tf.reduce_sum(masked_lm_weights) + 1e-6
          loss = numerator / denominator

          # Store some Tensorflow tensors for metrics calculation
          self.tf_preds = preds
          self.tf_per_example_loss = label_log_probs

          return loss

      self.loss_tensor = loss_fn
      
    return self.loss_tensor    

  # Function to get list of metric name for this model when perform training.
  def get_metric_names(self):
    return ['mlm_acc']

  # Function to get list of metric function for this model when perform training.
  def get_metric_functions(self):
 
    if self.accuracy_tensor is None:
      
      _, _, _, _, masked_lm_weights = self.get_input_tensors()
      _, _, log_probs, _ = self.get_mlm_output_tensors()

      def acc(y_true, y_pred):

        print('::y_true = ' + str(y_true))

        # y_true is IDs of masked tokens
        masked_lm_ids = y_true

        # y_pred is preds
        preds = y_pred

        acc_ = tf.metrics.accuracy(
          labels=tf.reshape(masked_lm_ids, [-1]),
          predictions=tf.reshape(preds, [-1]),
          weights=tf.reshape(masked_lm_weights, [-1]))

        return acc_

      self.accuracy_tensor = acc

    return [self.accuracy_tensor]


# Unit Test
print('-===================-')
print(__name__)
#if __name__ == '__main__':

if __name__ == '__main__' or __name__ == 'tensorflow.keras.initializers':

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
  itokens = BERTSentencePiecePretrainWrapper({'column_id': 0, "max_seq_length": 16}, data)
  otokens = BERTSentencePiecePretrainWrapper({'column_id': 1, "max_seq_length": 16}, data)

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
    'max_mask_tokens': 2,
    'cached_data_dir': '_cache_',
  }

  transformer = BERTWrapper(config, itokens, otokens)

  #data = transformer.load_encoded_data(data)
  #print(data)
  #exit(0)

  [input_tensors, output_tensors] = transformer.get_forward_tensors()
  print("=== INPUT_TENSOR ===")
  print(input_tensors)
  print("=== OUTPUT_TENSOR ===")
  print(output_tensors)
  model = Model(input_tensors, output_tensors)

  metric_funcs = transformer.get_metric_functions()

  from NLP_LIB.ext.bert.optimization import AdamWeightDecayOptimizer

  adamm = AdamWeightDecayOptimizer(learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-6,
    exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"]
  )

  model.compile(optimizer=adamm, 
    loss=transformer.get_loss_function(),
    metrics=transformer.get_metric_functions()
  )

  print(tf.trainable_variables())
  print(adamm.variables())
  print(len(tf.trainable_variables()))
  print(len(adamm.variables()))

  print(adamm.get_slot_names())
  print(len(adamm.get_slot_names()))  

  model.summary()

  input_ids = [[10, 14, 1, 18, 1]]
  input_mask = [[1, 1, 1, 1, 1]]
  token_type_ids = [[0, 0, 0, 1, 1]]
  masked_lm_positions = [[2, 4]]
  masked_lm_weights = [[1.0, 1.0]]
  masked_lm_ids = [[15, 20]]

  input_ids[0].extend([0 for _ in range(256 - len(input_ids[0]))])
  input_mask[0].extend([0 for _ in range(256 - len(input_mask[0]))])
  token_type_ids[0].extend([0 for _ in range(256 - len(token_type_ids[0]))])

  print(input_ids)

  print('Start unit testing')

  sess = K.get_session()
  init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
  sess.run(init)

  model.fit(x=[input_ids, input_mask, token_type_ids, masked_lm_positions, masked_lm_weights], y=[masked_lm_ids], batch_size=1, epochs=10,
    callbacks=[]
  )
  y = model.predict([input_ids, input_mask, token_type_ids, masked_lm_positions, masked_lm_weights])
  print(y)

  print('Finished.')
