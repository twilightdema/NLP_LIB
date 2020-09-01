import os
import sys
import math
import numpy as np
from NLP_LIB.nlp_core.dataset_wrapper import DatasetWrapper

class FederatedData(DatasetWrapper):

  def __init__(self, config, dataset, node_count, node_id):
    config['dataset_name'] = '_federated_' + str(node_count) + '_' + str(node_id) + '_' + dataset.config['dataset_name']
    super(FederatedData, self).__init__(config)
    self.dataset = dataset
    self.node_id = node_id
    self.node_count = node_count
    self.X_federated = None
    self.Y_federated = None
    self.X_Valid_federated = None
    self.Y_Valid_federated = None

  def _split_data(self, data, node_count, data_transform):
    data = np.array(data)
    data_shape = data.shape
    print('[INFO] Data Shape = ' + str(data_shape))
    if data_transform is None or data_transform.get_data_dimension() == 1:
      # If data is single series, we return list of splitted data
      data_count = data.shape[0]
      chunk_size = math.ceil(data_count / node_count)
      print('[INFO] Federated Data chunk size = ' + str(chunk_size))
      ret = [[] for _ in range(node_count)]
      for i in range(data_count):
        node = int(i / chunk_size)
        ret[node].append(data[i])
    else:
      # If data is multiple series, we return list of list of splitted data.
      series_count = data.shape[0]
      data_count = data.shape[1]
      chunk_size = math.ceil(data_count / node_count)
      print('[INFO] Federated Data chunk size = ' + str(chunk_size))
      ret = []
      for j in range(series_count):
        serie = [[] for _ in range(node_count)]
        for i in range(data_count):
          node = int(i / chunk_size)
          serie[node].append(data[j][i])
        ret.append(serie)
    return ret

  def simulate_federated_data(self, X, Y, X_Valid, Y_Valid, data_transform):
    # This function simulate federated data by divide data set in to 'node_count' chunks.
    print('[INFO] Begin simulate Federated data...')
    print('[INFO] Whole dataset train size = ' + str(len(X)))
    print('[INFO] Whole dataset valid size = ' + str(len(X_Valid)))    
    self.X_federated = self._split_data(X, self.node_count, data_transform)
    self.Y_federated = self._split_data(Y, self.node_count, data_transform)
    self.X_Valid_federated = self._split_data(X_Valid, self.node_count, data_transform)
    self.Y_Valid_federated = self._split_data(Y_Valid, self.node_count, data_transform)
    exit(0)

  # Get all data as list (probably for debug propose)
  def load_as_list(self):
    return self.dataset.load_as_list()

  # Perform post-processing on fully loaded data (Maybe filter or some custom logic on dataset setting)
  # In federated simulation, preaggregated data and loaded data is the whole data, so we load the splited data here.
  def postprocess_data_loading(self, X, Y, X_Valid, Y_Valid, data_transform):
    if self.X_federated is None:
      self.simulate_federated_data(X, Y, X_Valid, Y_Valid, data_transform)
    return (self.X_federated[self.node_id], self.Y_federated[self.node_id], self.X_Valid_federated[self.node_id], self.Y_Valid_federated[self.node_id])

  # Get unique data from the dataset as list
  def get_unique_data(self, column_id):
    return self.dataset.get_unique_data(column_id)

  # Get local directory that store data
  def get_local_data_dir(self):
    return self.dataset.get_local_data_dir()

  # Get token seperator (Ex. ' ' for English, '' for Thai - because no seperator)
  def get_trivial_token_separator(self):
    return self.dataset.get_trivial_token_separator()

# Unit Test
if __name__ == '__main__':
  from NLP_LIB.datasets import ColaDatasetWrapper
  dataset = ColaDatasetWrapper({'base_data_dir': 'tmp'})
  federated_data = FederatedData({'base_data_dir': 'tmp'}, dataset, 10, 0)
  x, y, x_valid, y_valid = dataset.postprocess_data_loading(federated_data.load_as_list(), None)
  print('X => ' + str(len(x)))
  print('Y => ' + str(len(y)))
  print('X_Valid => ' + str(len(x_valid)))
  print('Y_Valid => ' + str(len(y_valid)))
  print(x[0])
  print(y[0])

