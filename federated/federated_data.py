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
    #data = np.array(data)
    #data_shape = data.shape
    #print('[INFO] Data Shape = ' + str(data_shape))
    if data_transform is None or data_transform.get_data_dimension() == 1:
      # If data is single series, we return list of splitted data
      #data_count = data.shape[0]
      data_count = len(data)
      chunk_size = math.ceil(data_count / node_count)
      print('[INFO] Whole dataset size = ' + str(data_count))
      print('[INFO] Federated Data chunk size = ' + str(chunk_size))
      ret = [[] for _ in range(node_count)]
      for i in range(data_count):
        node = int(i / chunk_size)
        ret[node].append(data[i])
      ret = [np.array(s) for s in ret]
    else:
      # If data is multiple series, we return list of list of splitted data.
      #series_count = data.shape[0]
      #data_count = data.shape[1]
      series_count = len(data)
      data_count = len(data[0])
      chunk_size = math.ceil(data_count / node_count)
      print('[INFO] Series Count = ' + str(series_count))
      print('[INFO] Whole dataset size = ' + str(data_count))
      print('[INFO] Federated Data chunk size = ' + str(chunk_size))
      ret = []
      for j in range(series_count):
        serie = [[] for _ in range(node_count)]
        for i in range(data_count):
          node = int(i / chunk_size)
          serie[node].append(data[j][i])
        serie = [np.array(s) for s in serie]
        ret.append(serie)
    return ret

  def simulate_federated_data(self, X, Y, X_Valid, Y_Valid, data_transform, column_id):
    # This function simulate federated data by divide data set in to 'node_count' chunks.
    if column_id == 0:
      print('[INFO] Begin simulate Federated data (X)...')
      self.X_federated = self._split_data(X, self.node_count, data_transform)
      print('[INFO] Begin simulate Federated data (X_Valid)...')
      self.X_Valid_federated = self._split_data(X_Valid, self.node_count, data_transform)
    elif column_id == 1:
      print('[INFO] Begin simulate Federated data (Y)...')
      self.Y_federated = self._split_data(Y, self.node_count, data_transform)
      print('[INFO] Begin simulate Federated data (Y_Valid)...')
      self.Y_Valid_federated = self._split_data(Y_Valid, self.node_count, data_transform)

  # Get all data as list (probably for debug propose)
  def load_as_list(self):
    return self.dataset.load_as_list()

  # Perform post-processing on fully loaded data (Maybe filter or some custom logic on dataset setting)
  # In federated simulation, preaggregated data and loaded data is the whole data, so we load the splited data here.
  # Column ID is 0=input, 1=output, -1=both, because some postprocessing should ignore X or Y side those we are not interested in...
  def postprocess_data_loading(self, X, Y, X_Valid, Y_Valid, data_transform, column_id):
    if self.X_federated is None and column_id == 0:
      self.simulate_federated_data(X, Y, X_Valid, Y_Valid, data_transform, 0)
      return (self.X_federated[self.node_id], None, self.X_Valid_federated[self.node_id], None)
    if self.Y_federated is None and column_id == 1:
      self.simulate_federated_data(X, Y, X_Valid, Y_Valid, data_transform, 1)
      return (None, self.Y_federated[self.node_id], None, self.Y_Valid_federated[self.node_id])
    if self.X_federated is None and column_id == -1:
      self.simulate_federated_data(X, Y, X_Valid, Y_Valid, data_transform, 0)
      self.simulate_federated_data(X, Y, X_Valid, Y_Valid, data_transform, 1)
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

