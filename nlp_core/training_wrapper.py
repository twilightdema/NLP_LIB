from NLP_LIB.nlp_core.model_wrapper import ModelWrapper, SequenceModelWrapper, TrainableModelWrapper
from NLP_LIB.nlp_core.dataset_wrapper import DatasetWrapper
from NLP_LIB.optimizer.bert_optimizer import BERTOptimizer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from NLP_LIB.nlp_core.log_current_epoch_wrapper import LogCurrentEpochWrapper
import tensorflow as tf
import random, os, sys, re
import numpy as np

# Utility class for ModelCheckPoint that support Multi-GPU model extraction
class RefModelCheckpoint(ModelCheckpoint):
  def __init__(self, filepath, ref_model, **kwargs):
    """
    Additional keyword args are passed to ModelCheckpoint; see those docs for information on what args are accepted.
    :param filepath:
    :param alternate_model: Keras model to save instead of the default. This is used especially when training multi-
                            gpu models built with Keras multi_gpu_model(). In that case, you would pass the original
                            "template model" to be saved each checkpoint.
    :param kwargs:          Passed to ModelCheckpoint.
    """
    self.ref_model = ref_model
    super().__init__(filepath, **kwargs)

  def on_epoch_end(self, epoch, logs=None):
    model_before = self.model
    self.model = self.ref_model
    super().on_epoch_end(epoch, logs)
    self.model = model_before

# Base class for training a Trainable Model
class TrainingWrapper:

  def __init__(self, trainable_model, input_transform, output_transform, callback_list, training_config):
    # Check if instance of model is trainable
    print(type(trainable_model))
    print(isinstance(trainable_model, TrainableModelWrapper))
    if not isinstance(trainable_model, TrainableModelWrapper):
      raise ValueError('TraningWrapper can be used only with subclass of TrainableModelWrapper')
    # Check if training_config has all mandatory configuration values
    required_training_params = {
      'optimizer',
      'batch_size',
      'epochs',
      'watch_metric',
      'output_dir',
      'save_weight_history',
    }
    missing_training_params = [param for param in required_training_params if param not in training_config]
    if len(missing_training_params) > 0:
      raise ValueError('Missing training_config: ' + ','.join(missing_training_params))

    self.trainable_model = trainable_model
    self.training_config = training_config
    self.callback_list = callback_list
    self.input_transform = input_transform
    self.output_transform = output_transform

    self.multi_gpu = False    
    if 'multi_gpu' in training_config and training_config['multi_gpu']:
      self.multi_gpu = True

  def get_available_gpus(self):
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

  def train(self, dataset):
    # Transform data into format to be fed into model

    # Below code is more suitable for run mode than train mode
    '''
    (X, Y, X_valid, Y_valid) = dataset.load_as_list()
    X = self.trainable_model.encode_input(X)
    Y = self.trainable_model.encode_output(Y)
    X_valid = self.trainable_model.encode_input(X_valid)
    Y_valid = self.trainable_model.encode_output(Y_valid)
    '''

    # If using multi-gpu, then we save model/log files in other directory than normal one
    dir_suffix = ''
    gpu_count = 1
    if self.multi_gpu:
      gpu_count = len(self.get_available_gpus())
      # Changed to save multi-gpu model at the same path as single gpu model
      #if gpu_count > 1:
      #  dir_suffix = '_' + str(gpu_count) + 'gpus'
    print('Training on ' + str(gpu_count) + ' GPU(s)')

    # In case of train mode, we can load data in the wqay that we can utilize caching feature.
    # We separate call between input and output because they are use different transformation approach.
    (X, Y, X_valid, Y_valid) = self.trainable_model.load_encoded_data(dataset)
    '''
    xx = X[0:5]
    yy = Y[0:5]
    print('xx')
    print(xx)
    print('yy')
    print(yy)
    '''

    training_data_count = X.shape[0]
    print('Training data count = ' + str(training_data_count))
    batch_count = int(training_data_count / self.training_config['batch_size'])
    print('Batch count = ' + str(batch_count))
    training_data_count = int(batch_count * self.training_config['batch_size'])
    print('Training data used = ' + str(training_data_count))

    validation_data_count = X_valid.shape[0]
    print('Validation data count = ' + str(validation_data_count))
    batch_count = int(validation_data_count / self.training_config['batch_size'])
    print('Batch count = ' + str(batch_count))
    validation_data_count = int(batch_count * self.training_config['batch_size'])
    print('Validation data used = ' + str(validation_data_count))

    X = X[0:training_data_count]
    Y = Y[0:training_data_count]
    X_valid = X_valid[0:validation_data_count]
    Y_valid = Y_valid[0:validation_data_count]

    # If multi-model, wrap it as Data Parallel trainable model
    if gpu_count > 1:
      with tf.device('/cpu'):
        [input_tensors, output_tensors] = self.trainable_model.get_forward_tensors()
        print("=== INPUT_TENSOR ===")
        print(input_tensors)
        print("=== OUTPUT_TENSOR ===")
        print(output_tensors)
        model = Model(input_tensors, output_tensors)
      print("=== CPU TEMPLATE MODEL ===")
      model.summary()
      single_gpu_model = model # For saving weight
      model = multi_gpu_model(model, gpus=gpu_count)
      print("=== MULTI-GPU MODEL ===")
      model.summary()

    elif gpu_count == 1:
      with tf.device('/gpu'):
        [input_tensors, output_tensors] = self.trainable_model.get_forward_tensors()
      model = Model(input_tensors, output_tensors)
      single_gpu_model = model

    elif gpu_count == 0:
      with tf.device('/cpu'):
        [input_tensors, output_tensors] = self.trainable_model.get_forward_tensors()
      model = Model(input_tensors, output_tensors)
      single_gpu_model = model

    # Home of output directory (support multi-OS)
    output_dir = os.path.join(*re.split('/|\\\\', self.training_config['output_dir']))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # Callback to model before compile, init_from_checkpoint is loaded here.
    self.trainable_model.on_before_compile(single_gpu_model)

    # If resume training, load latest checkpoint
    # Checkpoint saving directory
    checkpoint_dir = os.path.join(output_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    last_checkpoint_filepath = os.path.join(checkpoint_dir, 'last_weight' + dir_suffix + '.h5')    
    current_epoch_wrapper = LogCurrentEpochWrapper(self.training_config, dir_suffix)
    initial_epoch = 0
    if 'resume_if_possible' in self.training_config and self.training_config['resume_if_possible'] == True:
      initial_epoch = current_epoch_wrapper.get_current_epoch()
      print('Init model ' + str(self) + ' from epoch: ' + str(initial_epoch))
      if os.path.exists(last_checkpoint_filepath):
        print('Init model ' + str(self) + ' from checkpoint: ' + last_checkpoint_filepath)
        single_gpu_model.load_weights(last_checkpoint_filepath)
    
    self.training_config['initial_epoch'] = initial_epoch

    optimizer = self.training_config['optimizer']
    if optimizer == 'adam':
      optimizer_params = self.training_config['optimizer_params']
      optimizer = Adam(optimizer_params[0], optimizer_params[1], optimizer_params[2], epsilon=optimizer_params[3])
    elif optimizer == 'bert_adam':
      optimizer_params = self.training_config['optimizer_params'] 
      # Calculate total step and set it to decay_steps (learning rate reachs 0 in the every end)
      total_steps = batch_count * self.training_config['epochs']
      print('[INFO] Training with BERT Optimizer with decay_steps = ' + str(total_steps))
      optimizer = BERTOptimizer(
        decay_steps = total_steps, # 100000,
        warmup_steps = optimizer_params[2], # 10000,
        learning_rate = optimizer_params[0], # 1e-4,
        weight_decay = optimizer_params[1], # 0.01,
        weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],                
      )
    
    # Add model metric names and tensors to tracking list
    metric_names = self.trainable_model.get_metric_names()
    metric_funcs = self.trainable_model.get_metric_functions()

    '''
    metric_names = self.trainable_model.get_metric_names()
    metric_tensors = self.trainable_model.get_metric_tensors()
    for metric_name, metric_tensor in zip(metric_names, metric_tensors):
      print('Add Metric: ' + metric_name)
      model.metrics_names.append(metric_name)
      model.metrics_tensors.append(metric_tensor)
    '''

    model.compile(optimizer=optimizer, 
      loss=self.trainable_model.get_loss_function(),
      metrics=metric_funcs
      )

    model.summary()

    # Also add gradient norm as a default metric
    # Get a "l2 norm of gradients" tensor
    def get_gradient_norm(model):
      with K.name_scope('gradient_norm'):
        grads = K.gradients(model.total_loss, model.trainable_weights)
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
      return norm
    model.metrics_names.append("gradient_norm")

    if (hasattr(model, 'metrics_tensors')):
      model.metrics_tensors.append(get_gradient_norm(model))      
    else:
      model.metrics.append(get_gradient_norm(model))      

    x_feed = [X]
    y_feed = [Y]

    x_valid_feed = [X_valid]
    y_valid_feed = [Y_valid]

    # If model is sequence model, we have to feed prev_output too.
    # TODO: Can we make embed the flow to generate input list into the model?
    if isinstance(self.trainable_model, SequenceModelWrapper):
      x_feed.append(Y)
      x_valid_feed.append(Y_valid)

      # Also, if we are running Sequence Model, output will be logits but label will be sparse value.
      # Keras loss function need label and output to be in same dimension, thus we need to convert label to dense value too.
      # The converson to Dense is done in custom loss funciton in the model, but be need to "prepare" addition dimension to sparse label.
      y_feed = [np.expand_dims(Y, axis=2)]
      y_valid_feed = [np.expand_dims(Y_valid, axis=2)]

    checkpoint_filepath = os.path.join(checkpoint_dir, 'best_weight' + dir_suffix + '.h5')
    model_saver = RefModelCheckpoint(checkpoint_filepath, single_gpu_model, save_best_only=True, save_weights_only=True)

    # Also always save lastest model for continue training
    last_model_saver = RefModelCheckpoint(last_checkpoint_filepath, single_gpu_model, save_best_only=False, save_weights_only=True)

    # Tensorboard log directory
    tboard_log_dir = os.path.join(output_dir, 'tboard_log' + dir_suffix)
    if not os.path.exists(tboard_log_dir):
      os.makedirs(tboard_log_dir)
    tboard_log_saver = TensorBoard(tboard_log_dir, write_graph=True, write_images=True)

    # For saving weight history along with accuracy in each epoch (May use a lot of disk)
    verbose_model_saver = None
    if self.training_config['save_weight_history']:
      verbose_log_dir = os.path.join(output_dir, 'weight_history' + dir_suffix)
      if not os.path.exists(verbose_log_dir):
        os.makedirs(verbose_log_dir)
      verbose_weight_history_filepath = os.path.join(verbose_log_dir, 'weights.{epoch:02d}-{' + self.training_config['watch_metric'] + ':.4f}.h5')
      verbose_model_saver = RefModelCheckpoint(verbose_weight_history_filepath, single_gpu_model, save_best_only=False, save_weights_only=True)

    # Construct all training callbacks
    training_callbacks = [model_saver, last_model_saver, tboard_log_saver]
    if verbose_model_saver is not None:
      training_callbacks.append(verbose_model_saver)
    if self.callback_list is not None:
      for callback in self.callback_list:
        training_callbacks.append(callback.get_keras_callback())

    # Save current epoch
    training_callbacks.append(current_epoch_wrapper.get_keras_callback())

    model.summary()
    print('Start training.')
    '''
    with tf.Session(config = tf.ConfigProto(log_device_placement = False, allow_soft_placement=False)) as sess:
      init = tf.global_variables_initializer()
      sess.run(init)      

      model.fit(x=x_feed, y=y_feed,
        batch_size=self.training_config['batch_size'],
        epochs=self.training_config['epochs'],
        validation_data=(x_valid_feed, y_valid_feed),
        callbacks=training_callbacks,
        initial_epoch=initial_epoch
      )
    '''
    model.fit(x=x_feed, y=y_feed,
      batch_size=self.training_config['batch_size'],
      epochs=self.training_config['epochs'],
      validation_data=(x_valid_feed, y_valid_feed),
      callbacks=training_callbacks,
      initial_epoch=initial_epoch
    )

    print('Finished training.')

    # Return trained model (single_gpu_model) and validation set as output.
    # They are used for further benchmarking like in federated training.
    return (single_gpu_model, x_valid_feed, y_valid_feed)

  # This method is used to trim token those are after </S>
  def trimTextIds(self, idsList):
    retList = []
    for ids in idsList:
      ret = []
      for id in ids:
        ret.append(id)
        if id == 3: # Stop word
          break
      retList.append(ret)
    return retList

  def create_serving_model(self):
    # Create Keras serving model and return the model object

    # If using multi-gpu, then we save model/log files in other directory than normal one
    dir_suffix = ''
    gpu_count = len(self.get_available_gpus())
    if self.multi_gpu:
      gpu_count = len(self.get_available_gpus())
      # Changed to save multi-gpu model at the same path as single gpu model
      #if gpu_count > 1:
      #  dir_suffix = '_' + str(gpu_count) + 'gpus'
    print('Running on ' + str(gpu_count) + ' GPU(s)')

    # If multi-model, wrap it as Data Parallel trainable model
    if gpu_count > 1:
      with tf.device('/cpu'):
        [input_tensors, output_tensors] = self.trainable_model.get_forward_tensors()
        print("=== INPUT_TENSOR ===")
        print(input_tensors)
        print("=== OUTPUT_TENSOR ===")
        print(output_tensors)
        model = Model(input_tensors, output_tensors)
      print("=== CPU TEMPLATE MODEL ===")
      model.summary()
      single_gpu_model = model # For saving weight
      model = multi_gpu_model(model, gpus=gpu_count)
      print("=== MULTI-GPU MODEL ===")
      model.summary()

    elif gpu_count == 1:
      with tf.device('/gpu'):
        [input_tensors, output_tensors] = self.trainable_model.get_forward_tensors()
      model = Model(input_tensors, output_tensors)
      single_gpu_model = model

    elif gpu_count == 0:
      with tf.device('/cpu'):
        [input_tensors, output_tensors] = self.trainable_model.get_forward_tensors()
      model = Model(input_tensors, output_tensors)
      single_gpu_model = model

    # Home of output directory (support multi-OS)
    output_dir = os.path.join(*re.split('/|\\\\', self.training_config['output_dir']))
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    # Callback to model before compile, init_from_checkpoint is loaded here.
    self.trainable_model.on_before_compile(single_gpu_model)

    # If load best weight from the training folder.
    checkpoint_dir = os.path.join(output_dir, 'checkpoint')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    best_checkpoint_filepath = os.path.join(checkpoint_dir, 'best_weight' + dir_suffix + '.h5')    
    if os.path.exists(best_checkpoint_filepath):
      print('Init model ' + str(self) + ' from checkpoint: ' + best_checkpoint_filepath)
      single_gpu_model.load_weights(best_checkpoint_filepath)

    # Construct all training callbacks
    training_callbacks = []
    if self.callback_list is not None:
      for callback in self.callback_list:
        training_callbacks.append(callback.get_keras_callback())

    model.summary()

    return model

  def predict(self, mode, sampling_algorithm, generation_count, input_mode, input_path, prev_output_path = None, model = None):
    # Extract beam num is applicable
    beam_num = 1
    if sampling_algorithm is not None and sampling_algorithm.startswith('beam'):
      beam_num = int(sampling_algorithm[4:])
      sampling_algorithm = 'beam'
    else:
      sampling_algorithm = 'argmax'

    if model is None:
      model = self.create_serving_model()
      
    # In case of prediction mode and str type input, we call to model to encode input for us directly.
    # In file mode, we can encode them in batch manner.
    X = []
    Y_prev = None
    if input_mode == 'str':
      print('Encode string mode')
      xx = self.trainable_model.encode_input([input_path.split(',')])
      X.append(xx[0])
      # If we are using beam search, then we allow only 1 input sequence,
      # but we need it K times to store top-K sequences
      if sampling_algorithm == 'beam':
        for j in range(beam_num - 1):
          X.append(xx[0].copy())
      X = np.array(X)
      if prev_output_path is not None:
        xx = self.trainable_model.encode_output([prev_output_path.split(',')])
        Y_prev.append(xx[0])
        # If we are using beam search, then we allow only 1 input sequence
        # but we need it K times to store top-K sequences
        if sampling_algorithm == 'beam':
          for j in range(beam_num - 1):
            Y_prev.append(xx[0].copy())
        Y_prev = np.array(Y_prev)

    elif input_mode == 'file':
      print('Encode file mode')
      X = []
      with open(input_path, 'r', encoding='utf-8') as fin:
        for line in fin:
          xx = self.trainable_model.encode_input([line.strip().split(' ')])
          X.append(xx[0])
          # If we are using beam search, then we allow only 1 input sequence,
          # but we need it K times to store top-K sequences
          if sampling_algorithm == 'beam':
            for j in range(beam_num - 1):
              X.append(xx[0].copy())
            break
      X = np.array(X)
      if prev_output_path is not None:
        Y_prev = []
        with open(prev_output_path, 'r', encoding='utf-8') as fin:
          for line in fin:
            xx = self.trainable_model.encode_input([line.strip().split(' ')])
            Y_prev.append(xx[0])
            # If we are using beam search, then we allow only 1 input sequence
            # but we need it K times to store top-K sequences
            if sampling_algorithm == 'beam':
              for j in range(beam_num - 1):
                Y_prev.append(xx[0].copy())
              break
        Y_prev = np.array(Y_prev)

    prediction_data_count = X.shape[0]
    print('Prediction data count = ' + str(prediction_data_count))
    batch_count = int(prediction_data_count / self.training_config['batch_size'])
    # Batch count cannot be 0!
    batch_count = max(1, batch_count)
    print('Batch count = ' + str(batch_count))
    prediction_data_count = int(batch_count * self.training_config['batch_size'])
    print('Prediction data used = ' + str(prediction_data_count))

    X = X[0:prediction_data_count]

    x_feed = [X]

    # If model is sequence model, we have to feed prev_output too.
    # TODO: Can we make embed the flow to generate input list into the model?
    if isinstance(self.trainable_model, SequenceModelWrapper):
      # Case of not specify Y_prev, we reuse X as previous state of Y
      if Y_prev is None:
        Y_prev = X
      x_feed.append(Y_prev)

    print(X)

    print('X = ' + str(X))
    print('Y_prev = ' + str(Y_prev))
    print('X.shape = ' + str(X.shape))
    if Y_prev is not None:
      print('Y_prev.shape = ' + str(Y_prev.shape))

    print('Start predicting.')

    # Store log probabilities of K initial sequence, initial values are 0
    beam_log_probs = None
    beam_input_usage = 1
    if sampling_algorithm == 'beam':
      beam_log_probs = [0.0] * beam_num

    # If generation count > 0 then we are generation sequence.
    # We will remove </S> from input in such case.
    if generation_count > 0:

      # Perform generation for N times
      for i in range(generation_count):
        
        print('Generation #' + str(i))
        last_token_position = []

        for each_x in x_feed:
          for x_i in each_x:
            pass_end = False
            for j in range(len(x_i)):
              if pass_end:
                x_i[j] = 0
              else:
                if x_i[j] == 3:
                  last_token_position.append(j)
                  #x_i[j] = 0
                  pass_end = True

        #print('X = ' + str(X))
        #print('Y_prev = ' + str(Y_prev))

        Y = model.predict(x=x_feed)
        
        # If we are using beam search, them we will store top K token with highest probabilities
        # at the position of the </S> in input
        if sampling_algorithm == 'beam':
          # Store next word to be and to which input_id, alond with its new jointed probability
          next_candidates = []
          for input_id, y, last_token_pos in zip(range(beam_input_usage), Y, last_token_position):
            # Get probability of each tokens
            token_probs = y[last_token_pos-1, :]
            # Calculate softmax
            max_p = np.max(token_probs)
            e_x = np.exp(token_probs - max_p)
            token_probs = e_x / np.sum(e_x)
            # Calculate log probability, so we can add them up along the sequence generation process
            token_probs = np.log(token_probs / (np.sum(token_probs, -1) + 1e-8)) # 1e-8 is for prevent numerical instability
            tokens_with_prob_sorted = sorted(list(enumerate(token_probs)), key=lambda x:x[-1], reverse=True)
            tokens_with_prob_sorted = tokens_with_prob_sorted[:beam_num]
            print("BEAM TOKEN PROB for input_id = " + str(input_id) + ", last_token_pos = " + str(last_token_pos-1))
            print(tokens_with_prob_sorted)
            for token_id, token_log_prob in tokens_with_prob_sorted:
              next_candidates.append((input_id, token_id, beam_log_probs[input_id] + token_log_prob))
          # We keep only best K candidates, sorted by jointed probability
          next_candidates.sort(key=lambda x:x[-1], reverse=True)
          next_candidates = next_candidates[:beam_num]
          print("BEST K BEAMS")
          print(next_candidates)
          # Update number of input beam slot we are using
          beam_input_usage = len(next_candidates)

          # Construct new input sequences from best K beams
          tmp_X = X.copy()
          for new_input_id, candidate in enumerate(next_candidates):            
            input_id, token_id, new_log_prob = candidate
            X[new_input_id] = tmp_X[input_id]
            X[new_input_id, last_token_position[input_id]] = token_id
            X[new_input_id, last_token_position[input_id] + 1] = 3 # Add new </S> to new latest position
            beam_log_probs[new_input_id] = new_log_prob

          X_input = self.trainable_model.decode_output(self.trimTextIds(X))

          # Y Output from the function should be sequence of output text prediction
          Y_output = self.trainable_model.decode_output(self.trimTextIds(X))
          print('updated input = ' + str(X_input))
          # Y ID Output should be ID of predicted sequence
          Y_id_max = X

        else:
          # If not using beam search, we just sample next token from ARGMAX...
          # Argmax output along last axis to get prediction class with maximum probability
          Y_id_max = np.argmax(Y, axis=-1)
          #print('Predicted output ids = ' + str(Y_id_max))

          #X_input = self.trainable_model.decode_output(self.trimTextIds(X))
          #Y_output = self.trainable_model.decode_output(self.trimTextIds(Y_id_max))
          #print('input = ' + str(X_input))
          #print('Predicted output = ' + str(Y_output))

          # Add predicted output token at the last_token_pos to each input (at last_token_pos too)
          # Y_prev (if there is) has same pointer as X so we only need to update them once.
          for x, y, last_token_pos in zip(X, Y_id_max, last_token_position):
            x[last_token_pos] = y[last_token_pos-1]
            print('ARGMAX at pos ' + str(last_token_pos-1) + ' = ' + str(y[last_token_pos-1]))
            x[last_token_pos + 1] = 3 # Add new </S> to new latest position
          X_input = self.trainable_model.decode_input(self.trimTextIds(X))

          # Y Output from the function should be sequence of output text prediction
          Y_output = self.trainable_model.decode_output(self.trimTextIds(X))
          # Y ID Output should be ID of predicted sequence
          Y_id_max = X

          print('updated input = ' + str(X_input))

    else:
      Y = model.predict(x=x_feed)
      # Argmax output along last axis to get prediction class with maximum probability
      Y_id_max = np.argmax(Y, axis=-1)
      print('Predicted output ids = ' + str(Y_id_max))

      X_input = self.trainable_model.decode_input(X)
      Y_output = self.trainable_model.decode_output(Y_id_max)
      print('input = ' + str(X_input))
      print('Predicted output = ' + str(Y_output))
    
    print('Finished predicting.')
    return (Y_output, Y_id_max, Y)

# Unit Test
if __name__ == '__main__':
  from datasets.array_dataset_wrapper import ArrayDatasetWrapper
  from models.transformer_wrapper import TransformerWrapper
  from transforms.fullword_dictionary_wrapper import FullWordDictionaryWrapper
  from callbacks.dynamic_lr_wrapper import DynamicLearningRateWrapper

  def GenSample():
    x = random.randint(0, 99999)
    y = hex(x);  x = str(x)
    return x, y

  X, Y = [], []
  X_valid, Y_valid = [], []
  for _ in range(100):
    x, y = GenSample()
    X.append(list(x))
    Y.append(list(y))
  for _ in range(10):
    x_valid, y_valid = GenSample()
    X_valid.append(list(x_valid))
    Y_valid.append(list(y_valid))

  dataset_loader = ArrayDatasetWrapper({'values': (
    X, Y,
    X_valid, Y_valid
  )})

  itokens = FullWordDictionaryWrapper({'column_id': 0}, dataset_loader)
  otokens = FullWordDictionaryWrapper({'column_id': 1}, dataset_loader)

  config = {
    'len_limit': 64,
    'd_model': 64,
    'd_inner_hid': 64,
    'n_head': 4,
    'd_k': 64,
    'd_v': 64,
    'layers': 2,
    'dropout': 0.1,
    'share_word_emb': False,
    'max_input_length': 256,
    'cached_data_dir': '_cache_',
  }
  training_config = {
    'optimizer': 'adam',
    'optimizer_params': [0.001, 0.9, 0.98, 1e-9],
    'batch_size': 20,
    'epochs': 10,
    'watch_metric': 'val_acc',
    'output_dir': '_test_output_',
    'save_weight_history': True,
    #'multi_gpu':True,
  }
  lr_config = {
    'd_model': 64,
    'warmup': 50,
  }
  transformer = TransformerWrapper(config, itokens, otokens)
  dynamic_lr = DynamicLearningRateWrapper(lr_config, training_config, transformer, dataset_loader, itokens, otokens)
  training = TrainingWrapper(transformer, itokens, otokens, [dynamic_lr], training_config)

  gpus = training.get_available_gpus()
  print(gpus)

  training.train(dataset_loader)
  print('Finish.')


'''
# Unit Test
if __name__ == '__main__':

  from datasets.array_dataset_wrapper import ArrayDatasetWrapper
  from models.transformer_wrapper import TransformerWrapper
  from transforms.fullword_dictionary_wrapper import FullWordDictionaryWrapper
  from callbacks.dynamic_lr_wrapper import DynamicLearningRateWrapper

  def GenSample():
    x = random.randint(0, 99999)
    y = hex(x);  x = str(x)
    return x, y

  X, Y = [], []
  X_valid, Y_valid = [], []
  for _ in range(100):
    x, y = GenSample()
    X.append(list(x))
    Y.append(list(y))
  for _ in range(10):
    x_valid, y_valid = GenSample()
    X_valid.append(list(x_valid))
    Y_valid.append(list(y_valid))

  dataset_loader = ArrayDatasetWrapper((
    X, Y,
    X_valid, Y_valid
  ))

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
  training_config = {
    'optimizer': 'adam',
    'batch_size': 20,
    'epochs': 10,
    'watch_metric': 'val_acc',
    'output_dir': 'tmp',
    'save_weight_history': True,
  }
  lr_config = {
    'd_model': 64,
    'warmup': 50,
  }
  transformer = TransformerWrapper(config, itokens, otokens)
  dynamic_lr = DynamicLearningRateWrapper(lr_config, transformer, itokens, otokens)
  training = TrainingWrapper(transformer, [dynamic_lr], training_config)
  training.train(dataset_loader)
  print('Finish.')
'''
