import os
import sys
import math
from NLP_LIB.nlp_core.dataset_wrapper import DatasetWrapper

class FederatedData(DatasetWrapper):

  def __init__(self, config, dataset, node_count, node_id):
    config['dataset_name'] = '_federated_' + str(node_count) + '_' + dataset.config['dataset_name']
    super(FederatedData, self).__init__(config)
    self.dataset = dataset
    self.node_id = node_id
    self.node_count = node_count

  def simulate_federated_data(self):
    # This function simulate federated data by divide data set in to 'node_count' chunks
    # and saved in cached directory with node_count annotation.
    X, Y, X_Valid, Y_Valid = self.dataset.load_as_list()
    
    X_federated = [[] for _ in range(self.node_count)]
    Y_federated = [[] for _ in range(self.node_count)]
    X_Valid_federated = [[] for _ in range(self.node_count)]
    Y_Valid_federated = [[] for _ in range(self.node_count)]

    chunk_size = math.ceil(len(X) / self.node_count)
    print('[INFO] Federated Data chunk size = ' + str(chunk_size))

    for i, (x, y) in enumerate(zip(X, Y)):
      node = int(i / chunk_size)
      X_federated[node].append(x)
      Y_federated[node].append(y)

    for i, (x, y) in enumerate(zip(X_Valid, Y_Valid)):
      node = int(i / chunk_size)
      X_Valid_federated[node].append(x)
      Y_Valid_federated[node].append(y)

    return X_federated, Y_federated, X_Valid_federated, Y_Valid_federated

  # Get all data as list (probably for debug propose)
  def load_as_list(self):
    X_federated, Y_federated, X_Valid_federated, Y_Valid_federated = self.simulate_federated_data()
    return (X_federated[self.node_id], Y_federated[self.node_id], X_Valid_federated[self.node_id], Y_Valid_federated[self.node_id])

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
  x, y, x_valid, y_valid = federated_data.load_as_list()
  print('X => ' + str(len(x)))
  print('Y => ' + str(len(y)))
  print('X_Valid => ' + str(len(x_valid)))
  print('Y_Valid => ' + str(len(y_valid)))
  print(x[0])
  print(y[0])

