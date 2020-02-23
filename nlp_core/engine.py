import importlib
import random
import sys
import json
import numpy as np
import codecs
import os
import tensorflow as tf
import keras

from NLP_LIB.nlp_core.predefined import ConfigMapper

# sys.stdout.reconfigure(encoding='utf-8') # Python 3.7 only
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

np.random.seed(0)

sys.path.append('.')

from NLP_LIB.nlp_core.training_wrapper import TrainingWrapper
from NLP_LIB.nlp_core.dataset_wrapper import DatasetWrapper
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper
from NLP_LIB.nlp_core.callback_wrapper import CallbackWrapper
from NLP_LIB.nlp_core.model_wrapper import ModelWrapper

# Main class for NLP Engine
class NLPEngine:

  def __init__(self):
    self.callbacks_module = importlib.import_module('NLP_LIB.callbacks')
    self.datasets_module = importlib.import_module('NLP_LIB.datasets')
    self.models_module = importlib.import_module('NLP_LIB.models')
    self.transforms_module = importlib.import_module('NLP_LIB.transforms')

  def run_train(self, config):
    dataset = config['dataset']
    dataset_class = dataset['class']
    dataset_config = dataset['config']
    dataset_class = getattr(self.datasets_module, dataset_class)
    dataset = dataset_class(dataset_config)

    input_transform = config['input_transform']
    input_transform_class = input_transform['class']
    input_transform_config = input_transform['config']
    input_transform_class = getattr(self.transforms_module, input_transform_class)
    input_transform = input_transform_class(input_transform_config, dataset)

    output_transform = config['output_transform']
    output_transform_class = output_transform['class']
    output_transform_config = output_transform['config']
    output_transform_class = getattr(self.transforms_module, output_transform_class)
    output_transform = output_transform_class(output_transform_config, dataset)

    model = config['model']
    model_class = model['class']
    model_config = model['config']
    model_class = getattr(self.models_module, model_class)
    model = model_class(model_config, input_transform, output_transform)

    execution = config['execution']
    execution_config = execution['config']

    callbacks_ = config['callbacks']
    callbacks = []
    for callback in callbacks_:
      callback_class = callback['class']
      callback_config = callback['config']
      callback_class = getattr(self.callbacks_module, callback_class)
      callback = callback_class(callback_config, execution_config, model, dataset, input_transform, output_transform)
      callbacks.append(callback)

    training = TrainingWrapper(model, input_transform, output_transform, callbacks, execution_config)
    training.train(dataset)

  def run_prediction(self, mode, sampling_algorithm, generation_count, config, input_mode, input_path):
    print('Running in ' + mode + ' mode for input_mode = ' + input_mode + ', input_path = ' + input_path)

    dataset = config['dataset']
    dataset_class = dataset['class']
    dataset_config = dataset['config']
    dataset_class = getattr(self.datasets_module, dataset_class)
    dataset = dataset_class(dataset_config)

    input_transform = config['input_transform']
    input_transform_class = input_transform['class']
    input_transform_config = input_transform['config']
    input_transform_class = getattr(self.transforms_module, input_transform_class)
    input_transform = input_transform_class(input_transform_config, dataset)

    output_transform = config['output_transform']
    output_transform_class = output_transform['class']
    output_transform_config = output_transform['config']
    output_transform_class = getattr(self.transforms_module, output_transform_class)
    output_transform = output_transform_class(output_transform_config, dataset)

    model = config['model']
    model_class = model['class']
    model_config = model['config']
    model_class = getattr(self.models_module, model_class)
    model = model_class(model_config, input_transform, output_transform)

    execution = config['execution']
    execution_config = execution['config']

    callbacks_ = config['callbacks']
    callbacks = []
    for callback in callbacks_:
      callback_class = callback['class']
      callback_config = callback['config']
      callback_class = getattr(self.callbacks_module, callback_class)
      callback = callback_class(callback_config, execution_config, model, dataset, input_transform, output_transform)
      callbacks.append(callback)

    training = TrainingWrapper(model, input_transform, output_transform, callbacks, execution_config)
    return training.predict(mode, sampling_algorithm, generation_count, input_mode, input_path)

  def run_server(self, config):
    print('Running server for model: ' + str(config))

    dataset = config['dataset']
    dataset_class = dataset['class']
    dataset_config = dataset['config']
    dataset_class = getattr(self.datasets_module, dataset_class)
    dataset = dataset_class(dataset_config)

    input_transform = config['input_transform']
    input_transform_class = input_transform['class']
    input_transform_config = input_transform['config']
    input_transform_class = getattr(self.transforms_module, input_transform_class)
    input_transform = input_transform_class(input_transform_config, dataset)

    output_transform = config['output_transform']
    output_transform_class = output_transform['class']
    output_transform_config = output_transform['config']
    output_transform_class = getattr(self.transforms_module, output_transform_class)
    output_transform = output_transform_class(output_transform_config, dataset)

    model = config['model']
    model_class = model['class']
    model_config = model['config']
    model_class = getattr(self.models_module, model_class)
    model = model_class(model_config, input_transform, output_transform)

    execution = config['execution']
    execution_config = execution['config']

    callbacks_ = config['callbacks']
    callbacks = []
    for callback in callbacks_:
      callback_class = callback['class']
      callback_config = callback['config']
      callback_class = getattr(self.callbacks_module, callback_class)
      callback = callback_class(callback_config, execution_config, model, dataset, input_transform, output_transform)
      callbacks.append(callback)

    session = keras.backend.get_session()
    graph = tf.get_default_graph()
    training = TrainingWrapper(model, input_transform, output_transform, callbacks, execution_config)
    serving_model = training.create_serving_model()

    from NLP_LIB.nlp_core.serving import ModelServer
    model_server = ModelServer(training, serving_model, graph, session, str(config))
    model_server.start_server()

    return 0
    # return training.predict(mode, sampling_algorithm, generation_count, input_mode, input_path)

def main(argv):
  if len(argv) < 2:
    print("Usage: python3 <APP_NAME>.py <CONFIG_FILE_PATH> <optional: train | predict> <optional: str:XXX | file:XXX>")
    exit(1)

  mode = 'train'
  if len(argv) > 2:
    mode = argv[2]
  print('mode = ' + mode)

  generation_count = 0
  sampling_algorithm = None
  if mode.startswith('generate:'):
    tokens = mode.split(':')
    generation_count = int(tokens[1])
    if len(tokens) > 2:
      sampling_algorithm = tokens[2]
    mode ='generate'
    print('Running generating mode with N = ' + str(generation_count) + ' using sampling algorithm: ' + str(sampling_algorithm))

  if (mode == 'predict' or mode == 'generate')and len(argv) < 4:
    print('Prediction / Generation mode require data source input in format str:XXX or file:XXX')
    exit(1)

  input_mode = None
  input_path = None
  output_path = None
  if mode == 'predict' or mode == 'generate':
    input_arg = argv[3] 
    input_mode = input_arg[:input_arg.find(':')] 
    print('input_mode = ' + input_mode) 
    if input_mode != 'file' and input_mode != 'str':
      print('Prediction / Generation mode require data source input in format str:XXX or file:XXX')
      exit(1)
    input_path = input_arg[input_arg.find(':') + 1 :]

    if len(argv) > 4:
      output_path = argv[4]
    else:
      output_path = '_outputs_/output.txt'

  config_path = argv[1]

  execution_config = None

  # If config file is not found, then we look into predefined shortcut map for the config file
  if not os.path.isfile(config_path):
    config_path = ConfigMapper.get_config_path_for(config_path)
    if config_path is None:
      # Try to generate config as per shortcut text
      execution_config = ConfigMapper.construct_json_config_for_shortcut(argv[1])
      if execution_config is None:
        print('Invalid run shortcut or JSON configure path.')
    else:
      dir_name = os.path.dirname(os.path.realpath(__file__))
      config_path = dir_name + '/../' + config_path

  if execution_config is None:
    with open(config_path, 'r', encoding='utf8') as json_file:  
      execution_config = json.load(json_file)  

  engine = NLPEngine()

  if mode == 'train':
    engine.run_train(execution_config)
  elif mode == 'predict' or mode == 'generate':
    (Y_output, Y_id_max, Y) = engine.run_prediction(mode, sampling_algorithm, generation_count, execution_config, input_mode, input_path)
    print('==== PREDICTION OUTPUT ====')
    print(Y_output)

    # Save output to file
    with open(output_path, 'w', encoding='utf-8') as fout:
      for output_entry in Y_output:
        fout.write(str(output_entry) + '\n')
    print('Output is written to: ' + output_path)
  elif mode == 'serve':
    # Running model in serve mode
    engine.run_server(execution_config)

  print('Finish.')

# Main entry point
if __name__ == '__main__':
  main(sys.argv)
