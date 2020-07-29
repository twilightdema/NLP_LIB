from NLP_LIB.nlp_core.model_wrapper import EncoderModelWrapper, DecoderModelWrapper, SequenceModelWrapper, TrainableModelWrapper
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

import importlib

# Base class for sequence transfer learning.
# We will initialize encoder side of network based on configuration, then wiring encoder output to finetune part of the model
class SequenceTransferLearningWrapper(EncoderModelWrapper, TrainableModelWrapper):

  def get_configuration_list(self):
    conf_list = super().get_configuration_list()
    conf_list.extend([
      {
        'name': 'output_class_num',
        'type': 'int',
        'default': 4,
        'required': True,
        'remark': 'Output class number'
      },
      {
        'name': 'encoder_model',
        'type': 'object',
        'default': {},
        'required': True,
        'remark': 'Configuration of model used as encoder'
      },
      {
        'name': 'encoder_dict_dataset',
        'type': 'object',
        'default': {},
        'required': True,
        'remark': 'Configuration of dataset the encoder was originally trained from. It is used to construct dictionary of input'
      },
      {
        'name': 'encoder_checkpoint',
        'type': 'string',
        'default': '',
        'required': True,
        'remark': 'Path to checkout of encoder to be initialized from'
      },
      {
        'name': 'train_encoder',
        'type': 'boolean',
        'default': False,
        'required': True,
        'remark': 'Config whether we allow training encoder part of model or not'
      },
      {
        'name': 'max_input_length',
        'type': 'int',
        'default': 256,
        'required': True,
        'remark': 'Maximum input length, Can be None if we are not running deterministic shape. Finetune on classification task normally need this value explicitly defined.'
      },
    ])
    return conf_list

  # Function to return list of objects to differentiate cached of input/output that model will use.
  # Basically it is configurations that effect encoded data.
  def get_data_effected_configs(self):
    # For transformer, max_input_lenth effect length of encoded input/output
    return 'D' + str(type(self.encoder_dict_dataset).__name__) + '_' + str(self.config['max_input_length']) + super(SequenceTransferLearningWrapper, self).get_data_effected_configs()

  # When initialize model, we check if all required configuration values
  # are available and issue error if not.
  def __init__(self, config, input_data_transform, output_data_transform):    
    super(SequenceTransferLearningWrapper, self).__init__(config, input_data_transform, output_data_transform)

    # For transfer learning, we will change input_data_transform to use configuration of language model dataset.
    # AKA, use dict from language model dataset instead of fine tuning dataset.
    # We use the trick of creating new instance of the same class of input_data_transform, but put language model
    # data set as a parameter.
    dataset_module = importlib.import_module('NLP_LIB.datasets')
    encoder_dict_dataset_class = self.config['encoder_dict_dataset']['class']
    encoder_dict_dataset_config = self.config['encoder_dict_dataset']['config']
    encoder_dict_dataset_class = getattr(dataset_module, encoder_dict_dataset_class)
    print('encoder_dict_dataset_class = ' + str(encoder_dict_dataset_class))
    encoder_dict_dataset = encoder_dict_dataset_class(encoder_dict_dataset_config)
    print('encoder_dict_dataset = ' + str(encoder_dict_dataset))
    input_data_transform_class = input_data_transform.__class__
    print('input_data_transform_class = ' + str(input_data_transform_class))
    lm_input_data_transform = input_data_transform_class(input_data_transform.config, encoder_dict_dataset)
    self.encoder_dict_dataset = encoder_dict_dataset
    self.input_data_transform = lm_input_data_transform

    # Import and initialize encoder model as specified in configuraiton
    models_module = importlib.import_module('NLP_LIB.models')
    encoder_model_class = self.config['encoder_model']['class']
    encoder_model_config = self.config['encoder_model']['config']
    encoder_model_class = getattr(models_module, encoder_model_class)
    self.encoder_model = encoder_model_class(encoder_model_config, lm_input_data_transform, lm_input_data_transform)

    print('INIT Encoder: ')
    print(self.encoder_model)

    self.input_tensor = None
    self.output_tensor = None
    self.encoder_output_tensor = None
    self.loss_tensor = None
    self.label_tensor = None
    self.accuracy_tensor = None
    self.supervised_layer = None

  # Function to get Keras input tensors
  def get_input_tensors(self):
    if self.input_tensor is None:
      self.input_tensor = self.encoder_model.get_input_tensors()
    return self.input_tensor

  # Function to get Keras output tensors
  def get_output_tensors(self):
    if self.output_tensor is None:
      encoder_output_tensor = self.get_encoder_output_tensors()
      encoder_shape = encoder_output_tensor.get_shape().as_list()
      print('Encoder Output Shape = ' + str(encoder_shape))

      # Input shape
      input_tensor = self.get_input_tensors()
      input_tensor  = Lambda(lambda x:x[:,0:encoder_shape[1]])(input_tensor) # Filter out to have matched input and output length
      input_shape = input_tensor.get_shape().as_list()
      print('Input Shape = ' + str(input_shape))

      def get_hidden_of_clf(args):
        input_tensor, output_tensor = args
        print('input_tensor = ' + str(input_tensor))
        print('output_tensor = ' + str(output_tensor))

        #input_tensor = tf.Print(input_tensor, ['input_tensor', tf.shape(input_tensor), input_tensor], summarize=32)
        #output_tensor = tf.Print(output_tensor, ['output_tensor', tf.shape(output_tensor), output_tensor[0,5]], summarize=32)

        flatten_input = tf.reshape(input_tensor,[-1])
        flatten_encoder_output = tf.reshape(output_tensor,[-1, self.config['encoder_model']['config']['d_model']])
  
        print('flatten_input = ' + str(flatten_input))
        print('flatten_encoder_output = ' + str(flatten_encoder_output))

        # Find position of end_id in input, we will use it as signal for classification
        eq_tf = tf.equal(flatten_input, self.input_data_transform.config['clf_id'])
        print(eq_tf)
        #eq_tf = tf.Print(eq_tf, ['eq_tf', eq_tf])
        print(flatten_input)
        clf_pos = tf.where(eq_tf)
        #clf_pos = tf.Print(clf_pos, ['clf_pos', tf.shape(clf_pos), clf_pos], summarize=32)
        print(clf_pos)

        # Get hidden structure of the token that corresponding to end_id in input
        clf_hidden = tf.gather(flatten_encoder_output, clf_pos)
        #clf_hidden = tf.Print(clf_hidden, ['clf_hidden', tf.shape(clf_hidden), clf_hidden], summarize=32)
        return clf_hidden

      #input_tensor = Lambda(lambda x: tf.Print(x, ['INPUT:', x, tf.shape(x)]))(input_tensor)
      #encoder_output_tensor = Lambda(lambda x: tf.Print(x, ['ENCODER OUTPUT:', x, tf.shape(x)]))(encoder_output_tensor)
      clf_hidden = Lambda(get_hidden_of_clf)([input_tensor, encoder_output_tensor])
      print('clf_hidden = ' + str(clf_hidden))
      #clf_hidden = Lambda(lambda x: tf.Print(x, ['CLF_HIDDEN:', x, tf.shape(x)]))(clf_hidden)

      # Flatten clf_hidden list    
      clf_hidden_shape = clf_hidden.get_shape().as_list()
      print('clf_hidden Shape = ' + str(clf_hidden_shape))
      flatten_dim = 1
      for dim in clf_hidden_shape:
        if dim is not None:
          flatten_dim = flatten_dim * dim
      print('clf_hidden flatten shape = ' + str(flatten_dim))
      flatten_clf_hidden = Reshape([flatten_dim])(clf_hidden)

      #flatten_clf_hidden = Lambda(lambda x: tf.Print(x, ['CLF_HIDDEN_FLAT:', x, tf.shape(x)]))(flatten_clf_hidden)
      print('flatten_clf_hidden = ' + str(flatten_clf_hidden))

      # Add fully connect layer as output layer
      print('Adding Softmax of output class num = ' + str(self.config['output_class_num']))
      softmax_layer = Dense(self.config['output_class_num'], use_bias=True, kernel_regularizer=regularizers.l2(l = 0.001), name='supervised_output')
      softmax_output = softmax_layer(flatten_clf_hidden)
      self.supervised_layer = softmax_layer

      # Add regularity, if specigfied
      if 'drop_out' in self.config:
        drop_out_rate = self.config['drop_out']
        print('Add Dropout with rate = ' + str(drop_out_rate))
        softmax_output = Dropout(drop_out_rate)(softmax_output)

      self.output_tensor = softmax_output     
    return self.output_tensor

  # Function to get other tensors those are specific to each model. Result map from name to tensor object.
  def get_immediate_tensors(self):
    return [] # self.encoder_model.get_immediate_tensors()

  # Function to encode input based on model configuration
  def encode_input(self, input_tokens):
    return self.encoder_model.encode_input(input_tokens)

  # Function to decode output based on model configuration
  def decode_output(self, output_vectors):
    return self.output_data_transform.decode(output_vectors)

  # Callback activated when just before model being compile.
  # Keras model is passed so We can use it to load saved weight from checkpoint.
  def on_before_compile(self, model):

    # Load language model checkpoint
    if 'encoder_checkpoint' in self.config and self.config['encoder_checkpoint'] is not None:
      # TODO: Can we load only partial of language model weights (Encoder side only?)
      # The language model checkpoint has been saved as entire model.
      # So we need to contruct entire, model than load its weights from checkpoint, but we will use only encoder side
      checkpoint_path = self.config['encoder_checkpoint']
      print('Init encoder model ' + str(self) + ' from checkpoint: ' + checkpoint_path)
      [lm_input_tensors, lm_output_tensors] = self.encoder_model.get_forward_tensors()
      full_language_model = Model(lm_input_tensors, lm_output_tensors)
      full_language_model.load_weights(checkpoint_path)

    # If there are fine tuning checkpoint specified, we override it over language model checkpoint
    if 'init_from_checkpoint' in self.config and self.config['init_from_checkpoint'] is not None:
      checkpoint_path = self.config['init_from_checkpoint']
      print('Init model ' + str(self) + ' from checkpoint: ' + checkpoint_path)
      model.load_weights(checkpoint_path)

    # If we are not allow training on other layers apart from output layer
    if 'train_encoder' in self.config and self.config['train_encoder'] == False:
      print('Freeze all weights except the additional output Layer...')
      for l in model.layers:
        if l == self.supervised_layer:
          print('Found Supervised Layer')
          l.trainable = True
          print(l.name, l.trainable)
        else:
          l.trainable = False      

  #############################################################
  ## Subclass for EncoderModelWrapper  
  # Function to return Keras Tensors of encoder output.
  def get_encoder_output_tensors(self):
    return self.encoder_model.get_encoder_output_tensors()

  #############################################################
  ## Subclass for ModelWrapper  
  # Function to contruct Keras tensors for running model in forward loop.
  # Return list of [inputs, outputs] tensors.
  def get_forward_tensors(self):

    # Predict prev_output shifted left plus additional new token from Decoder Output
    input_tensor = self.get_input_tensors()
    output_tensor = self.get_output_tensors()

    return [[input_tensor], [output_tensor]]

  # Function to return loss function for the model
  def get_loss_function(self):
    if self.loss_tensor is None:
      def get_loss(y_true, y_pred):
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
        loss = K.mean(loss)
        return loss
      self.loss_tensor = get_loss
    return self.loss_tensor   

  # Function to get list of metric name for this model when perform training.
  def get_metric_names(self):
    return ['acc']

  # Function to get list of metric function for this model when perform training.
  def get_metric_functions(self):
    if self.accuracy_tensor is None:
      def acc(y_true, y_pred):
        #y_true = tf.Print(y_true, ['y_true', tf.shape(y_true), y_true], summarize=32)
        #y_pred = tf.Print(y_pred, ['y_pred', tf.shape(y_pred), y_pred], summarize=32)
        corr = K.cast(K.equal(K.cast(K.argmax(y_true, axis=-1), 'int32'), K.cast(K.argmax(y_pred, axis=-1), 'int32')), 'float32')
        #corr = tf.Print(corr, ['y_corr', tf.shape(corr), corr], summarize=32)
        mean_corr = K.mean(corr)
        #mean_corr = tf.Print(mean_corr, ['mean_corr', tf.shape(mean_corr), mean_corr], summarize=32)
        return mean_corr
      self.accuracy_tensor = acc

    return [self.accuracy_tensor]
