import importlib
import random
import sys
import json
import numpy as np
import codecs
import os
import tensorflow as tf
import tensorflow.keras
import re
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from tensorflow.keras import backend as K

from NLP_LIB.nlp_core.predefined import ConfigMapper
from NLP_LIB.federated.federated_data import FederatedData

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

    # Detect if finetuing from multiple pretrain checkpoints.
    multiple_init_checkpoint_names = None
    multiple_init_checkpoints = None
    if 'model' in config and 'config' in config['model'] and 'encoder_checkpoint' in config['model']['config']:
      encoder_checkpoint = config['model']['config']['encoder_checkpoint']
      if os.path.isdir(encoder_checkpoint):
        multiple_init_checkpoint_names = os.listdir(encoder_checkpoint)
        multiple_init_checkpoints = list(map(lambda x: os.path.join(encoder_checkpoint, x), multiple_init_checkpoint_names))
        print('[INFO] Init from multiple checkpoints: ' + str(multiple_init_checkpoints))

    if multiple_init_checkpoints is None:
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

    else:

      execution = config['execution']
      execution_config = execution['config']
      base_output_dir = os.path.join(*re.split('/|\\\\', execution_config['output_dir']))

      for encoder_checkpoint, checkpoint_name in zip(multiple_init_checkpoints, multiple_init_checkpoint_names):

        config['model']['config']['encoder_checkpoint'] = encoder_checkpoint
        print('[INFO] Init from checkpoint: ' + str(encoder_checkpoint))

        # Save output to separated directory
        output_dir = os.path.join(base_output_dir, 'trials', checkpoint_name)
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        execution_config['output_dir'] = output_dir

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


  def run_train_federated_simulation(self, config, node_count):
    print('[INFO] Start running federated training simulation on ' + str(node_count) + ' node(s).')

    # Load some parameters of execution_config as they may have to be instruments
    execution = config['execution']
    execution_config = execution['config']    
    base_output_dir = os.path.join(execution_config['output_dir'], 'ftrain_' + str(node_count))
    if not os.path.exists(base_output_dir):
      os.makedirs(base_output_dir)
    base_epoch = execution_config['epochs']

    # The process of federated simulation is that we will train model on each node epoch-by-epoch.
    # After each epoch we will load trained model of each node and perform federated averaging on their weights.
    # We then save averaged weights to latest checkpoint of model in each node and proceed to next epoch.

    # Tensorboard log directory
    dir_suffix = '' # We do not use gpu_count in save path anymore
    tboard_log_dir = os.path.join(base_output_dir, 'tboard_log' + dir_suffix)
    if not os.path.exists(tboard_log_dir):
      os.makedirs(tboard_log_dir)
    log_writer = tf.summary.FileWriter(tboard_log_dir)

    for epoch in range(base_epoch):
      print('[INFO] Federated training epoch: ' + str(epoch))

      # Avoid memory leakage in Tensorflow / Keras
      K.clear_session()    

      federated_weights_list = []
      federated_model = None
      x_valid_feed = None
      y_valid_feed = None
      metric_names = None

      for node_id in range(node_count):
        print('[INFO] Running epoch ' + str(epoch) + ' of node: ' + str(node_id))
        dataset = config['dataset']
        dataset_class = dataset['class']
        dataset_config = dataset['config']
        dataset_class = getattr(self.datasets_module, dataset_class)
        dataset = dataset_class(dataset_config)

        federated_dataset = FederatedData(config, dataset, node_count, node_id) 

        input_transform = config['input_transform']
        input_transform_class = input_transform['class']
        input_transform_config = input_transform['config']
        input_transform_class = getattr(self.transforms_module, input_transform_class)
        input_transform = input_transform_class(input_transform_config, federated_dataset)

        output_transform = config['output_transform']
        output_transform_class = output_transform['class']
        output_transform_config = output_transform['config']
        output_transform_class = getattr(self.transforms_module, output_transform_class)
        output_transform = output_transform_class(output_transform_config, federated_dataset)

        model = config['model']
        model_class = model['class']
        model_config = model['config']
        model_class = getattr(self.models_module, model_class)
        model = model_class(model_config, input_transform, output_transform)

        # Change epoch to let each node train incrementally epoch-by-epoch
        execution_config['epochs'] = (epoch + 1) 

        # Change output directory to be include node_id so we save model from each node separately
        execution_config['output_dir'] = os.path.join(*re.split('/|\\\\', base_output_dir), 'federated_' + str(node_id))

        callbacks_ = config['callbacks']
        callbacks = []
        for callback in callbacks_:
          callback_class = callback['class']
          callback_config = callback['config']
          callback_class = getattr(self.callbacks_module, callback_class)
          callback = callback_class(callback_config, execution_config, model, federated_dataset, input_transform, output_transform)
          callbacks.append(callback)

        training = TrainingWrapper(model, input_transform, output_transform, callbacks, execution_config)        
        federated_model, x_valid, y_valid = training.train(federated_dataset)

        # Store validation data for used in federated evaluation
        if x_valid_feed is None or y_valid_feed is None:
          x_valid_feed = x_valid
          y_valid_feed = y_valid
        else:
          for i, (x, xn) in enumerate(zip(x_valid_feed, x_valid)):
            x_valid_feed[i] = np.append(x, xn, 0)
          for i, (y, yn) in enumerate(zip(y_valid_feed, y_valid)):
            y_valid_feed[i] = np.append(y, yn, 0)

        metric_names = training.trainable_model.get_metric_names()

        federated_weights = federated_model.get_weights()
        federated_weights_list.append(federated_weights)

      # [TODO]: Perform federated averaging on model weights
      print('[INFO] Finished federated training of epoch: ' + str(epoch))
      new_weights = list()

      print('[INFO] Perform federated averaging for epoch: ' + str(epoch))
      for weights_list_tuple in zip(*federated_weights_list):
          new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
      federated_model.set_weights(new_weights)
      # Save the averaged weight to center checkpoint
      checkpoint_dir = os.path.join(base_output_dir, 'checkpoint')
      if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
      dir_suffix = '' # We do not use gpu_count in save path anymore
      last_checkpoint_filepath = os.path.join(checkpoint_dir, 'last_weight' + dir_suffix + '.h5')
      print('[INFO] Saving averaged weight at: ' + last_checkpoint_filepath)
      federated_model.save_weights(last_checkpoint_filepath)

      # Perform averaged model evaluation here!
      print('Evaluate with federated evaluation dataset of size: ' + str(x_valid_feed[0].shape))
      metrics = federated_model.evaluate(
        x=x_valid_feed, y=y_valid_feed,
        batch_size=execution_config['batch_size']
      )

      summary_vals = [tf.Summary.Value(tag="loss", simple_value=metrics[0])]
      for i in range(len(metric_names)):
        summary_vals.append(tf.Summary.Value(tag=metric_names[i], simple_value=metrics[i + 1]))
      summary = tf.Summary(value=summary_vals)      
      log_writer.add_summary(summary, epoch)
      log_writer.flush()

      print('==== FEDETATED EVALUATION RESULTS ====')
      print(metrics)

      # Also save the averaged model to every federated node. This is equal to serialize and send updated model to every node.
      for node_id in range(node_count):
        output_dir = os.path.join(*re.split('/|\\\\', base_output_dir), 'federated_' + str(node_id))
        checkpoint_dir = os.path.join(output_dir, 'checkpoint')
        dir_suffix = '' # We do not use gpu_count in save path anymore
        last_checkpoint_filepath = os.path.join(checkpoint_dir, 'last_weight' + dir_suffix + '.h5')
        print('[INFO] Saving averaged weight at node ' + str(node_id) + ': ' + last_checkpoint_filepath)
        federated_model.save_weights(last_checkpoint_filepath)


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
  elif mode.startswith('ftrain:'):
    node_count = int(mode[len('ftrain:'):])
    print('[INFO] Perform Federated Training Simulation on ' + str(node_count) + ' node(s).')
    engine.run_train_federated_simulation(execution_config, node_count)
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
