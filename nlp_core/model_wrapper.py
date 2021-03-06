import h5py
import os
import sys
import re

# Base class for all DL models those can train with Back Propagation
class ModelWrapper:

  # This function list all parameter this model can support.
  # Each parameter is expected to be as in format shown in below example code.
  def get_configuration_list(self):
    return [
      {
        'name': 'cached_data_dir',
        'type': 'string',
        'default': '_cached_data_',
        'required': True,
        'remark': 'Directory for data cache.'
      }
    ]

  # When initialize model, we check if all required configuration values
  # are available and issue error if not.
  def __init__(self, config, input_data_transform, output_data_transform):
    # Detect all missing required configs and return error if there is
    required_config_list = [req['name'] for req in self.get_configuration_list() if req['required']]
    missing_config = [miss for miss in required_config_list if miss not in config]
    if len(missing_config) > 0:
      err_msg = 'Missing required configuration: ' + ','.join(missing_config)
      raise ValueError(err_msg)
    # Detect all missing optional configs and fill with default values
    optional_config_list = [(req['name'], req['default']) for req in self.get_configuration_list() if req['required'] == False]
    missing_optional_configs = [miss for miss in optional_config_list if miss[0] not in config]
    for missing_optional_config in missing_optional_configs:
      print('[WARNING]: Missing optional config: ' + missing_optional_config[0] + ', use default value: ' + str(missing_optional_config[1]))
      config[missing_optional_config[0]] = missing_optional_config[1]

    self.config = config
    self.input_data_transform = input_data_transform
    self.output_data_transform = output_data_transform

  # Function to get Keras input tensors
  def get_input_tensors(self):
    return None

  # Function to get Keras preprocessed input tensors
  def get_preprocessed_input_tensors(self):
    if self.input_data_transform.is_data_dynamically_aggregated():
      return self.input_data_transform.get_dynamically_aggregation_layer(self.get_input_tensors())
    else:
      return self.get_input_tensors()

  # Function to get Keras output tensors
  def get_output_tensors(self):
    return None

  # Function to get postprocessed Keras output tensors
  def get_postprocessed_output_tensors(self):
    # TODO: Do we really need postprocessed on model output?
    # Because we can do custom processing on Label before entering loss calculation, this may not needed.
    # We moved the dynamic aggregation to be performed on Label instead of here.
    '''
    if self.output_data_transform.is_data_dynamically_aggregated():
      return self.output_data_transform.get_dynamically_aggregation_layer(self.get_output_tensors())
    else:
      return self.get_output_tensors()
    '''
    return self.get_output_tensors()

  # Function to contruct Keras tensors for running model in forward loop.
  # Return list of [inputs, outputs] tensors.
  def get_forward_tensors(self):
    return None

  # Function to encode input based on model configuration
  def encode_input(self, input_tokens):
    return self.input_data_transform.encode(input_tokens)

  # Function to decode input based on model configuration
  def decode_input(self, input_vectors):
    return self.input_data_transform.decode(input_vectors)

  # Function to decode output based on model configuration
  def decode_output(self, output_vectors):
    return self.output_data_transform.decode(output_vectors)

  # Callback activated when just before model being compile.
  # Keras model is passed so We can use it to load saved weight from checkpoint.
  def on_after_init(self, model):
    if 'init_from_checkpoint' in self.config and self.config['init_from_checkpoint'] is not None:
      checkpoint_path = self.config['init_from_checkpoint']
      print('Init model ' + str(self) + ' from checkpoint: ' + checkpoint_path)
      model.load_weights(checkpoint_path)

# More specialized class for models those have encoder part
class EncoderModelWrapper(ModelWrapper):
  # Function to return Keras Tensors of encoder output.
  def get_encoder_output_tensors(self):
    return None

# More specialized class for models those have decoder part
class DecoderModelWrapper(ModelWrapper):
  # Function to return Keras Tensors of encoder output.
  def get_decoder_output_tensors(self):
    return None

# More specialized class for models those are sequence model.
# They will have previous output tensor as another input to the model.
class SequenceModelWrapper(ModelWrapper):
  # Function to return Keras Tensors of output at last time step (to be use as an input to model at this time step).
  def get_prev_output_tensors(self):
    return None

# More specialized class for models those are trainable.
# In order to be trainable, the model has to has tensor for loss and label value so we can minimize loss against it.
class TrainableModelWrapper(ModelWrapper):
  # Function to encode input based on model configuration
  def encode_output(self, output_tokens):
    return self.output_data_transform.encode(output_tokens)

  # Function to return function that calculate loss tensor of the model.
  def get_loss_function(self):
    return None
  # Function to get list of metric name for this model when perform training.
  def get_metric_names(self):
    return []
  # Function to get list of function to calculate metric tensor for this model when perform training.
  def get_metric_functions(self):
    return []

  # Function to get other tensors those are specific to each model. Result map from name to tensor object.
  def get_immediate_tensors(self):
    return {}

  # Function to return list of objects to differentiate cached of input/output that model will use.
  # Basically it is configurations that effect encoded data.
  def get_data_effected_configs(self):
    return self.input_data_transform.get_data_effected_configs() + self.output_data_transform.get_data_effected_configs()

  # When reading data from .h5 file, the data is automatically in numpy array format.
  # For some model, we will need to retain original format before caching (list of numpy array).
  # We use this function to identify if we should convert the cached data back to list or not
  def should_convert_cached_data_to_list(self):
    return False

  # Function to load and encode data from a dataset, based on model configuration, we can implement cache loading here.
  # The function should return (X, Y, X_valid, Y_valid) of encoded data.
  # In case of BERT or some model, the data can be already transformed by input/output transform class.
  # We can just load data from transformed data, otherwise we will load each data row and call to transformed class to preprocess them one-by-one.
  def load_encoded_data(self, dataset):

    X = None
    Y = None
    X_valid = None
    Y_valid = None

    # We give ability to have dataset perform "postprocessing" on pre-aggregated data here.
    # For loading pre-aggregated data, we pass data transformation instance as a parameter so that dataset can query for information of the transformations
    # those already applied to the data. (Ex. data shape and dimension of transformation output).
    if self.input_data_transform.is_data_preaggregated():
      X, _, X_valid, _ = dataset.postprocess_data_loading(*self.input_data_transform.load_preaggregated_data(), self.input_data_transform, 0)
    if self.output_data_transform.is_data_preaggregated():
      _, Y, _, Y_valid = dataset.postprocess_data_loading(*self.output_data_transform.load_preaggregated_data(), self.output_data_transform, 1)

    if X is None or Y is None:

      # Home of cached data directory (support multi-OS)
      cached_data_dir = os.path.join(*re.split('/|\\\\', self.config['cached_data_dir']))
      if not os.path.exists(cached_data_dir):
        os.makedirs(cached_data_dir)

      # Path for input data cache
      cached_data_dir_in = os.path.join(cached_data_dir, type(self.input_data_transform).__name__)    
      if X is None:
        if not os.path.exists(cached_data_dir_in):
          os.makedirs(cached_data_dir_in)
        cached_data_path_in = os.path.join(cached_data_dir_in, dataset.config['dataset_name'] + '_' + str(self.get_data_effected_configs()) + "_in.h5")

      # Path for output data cache
      cached_data_dir_out = os.path.join(cached_data_dir, type(self.output_data_transform).__name__)    
      if Y is None:
        if not os.path.exists(cached_data_dir_out):
          os.makedirs(cached_data_dir_out)
        cached_data_path_out = os.path.join(cached_data_dir_out, dataset.config['dataset_name'] + '_' + str(self.get_data_effected_configs()) + "_out.h5")

      print('Use caching data at: ' + str(cached_data_path_in) + ', ' + str(cached_data_path_out))

      X_ = None
      Y_ = None
      X_valid_ = None
      Y_valid_ = None

      # We give ability to have dataset perform "postprocessing" on fully loaded data here.
      # For loading directly from dataset, we pass data transformation as None because the data has not been transform yet.
      if (X is None and not os.path.exists(cached_data_path_in)) or (Y is None and not os.path.exists(cached_data_path_out)):
        (X_, Y_, X_valid_, Y_valid_) = dataset.postprocess_data_loading(*dataset.load_as_list(), None, -1)

      if X is None:
        if os.path.exists(cached_data_path_in):
          print('Loading input data from cache: ' + cached_data_path_in)
          with h5py.File(cached_data_path_in) as dfile:
            X, X_valid = dfile['X'][:], dfile['X_valid'][:]
        else:
          print('Loading input data from raw file and generate cached files...')
          X = self.encode_input(X_)
          X_valid = self.encode_input(X_valid_)
          with h5py.File(cached_data_path_in, 'w') as dfile:
            dfile.create_dataset('X', data=X)
            dfile.create_dataset('X_valid', data=X_valid)

      if Y is None:
        if os.path.exists(cached_data_path_out):
          print('Loading output data from cache: ' + cached_data_path_out)
          with h5py.File(cached_data_path_out) as dfile:
            Y, Y_valid = dfile['Y'][:], dfile['Y_valid'][:]
        else:
          print('Loading output data from raw file and generate cached files...')
          Y = self.encode_output(Y_)
          Y_valid = self.encode_output(Y_valid_)
          with h5py.File(cached_data_path_out, 'w') as dfile:
            dfile.create_dataset('Y', data=Y)
            dfile.create_dataset('Y_valid', data=Y_valid)

    #print(X)
    #exit(0)
    return (X, Y, X_valid, Y_valid)
