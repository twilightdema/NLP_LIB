import urllib.request
import sys, os, re

from nlp_core.dataset_wrapper import DatasetWrapper

# Subclass of DatasetWrapper that wrap reading file from Local Directory
class LocalDatasetWrapper(DatasetWrapper):

  # Initialize from local directory
  def __init__(self, config):
    super(LocalDatasetWrapper, self).__init__(config)  
    dataset_name = config['dataset_name']
    local_data_dir = config['local_src_dir']
    local_train_data_path = os.path.join(local_data_dir, 'train.txt')
    local_validation_data_path = os.path.join(local_data_dir, 'valid.txt')
    local_dict_path_prefix = os.path.join(local_data_dir, 'dict')

    self.local_data_dir = local_data_dir
    self.local_train_data_path = local_train_data_path
    self.local_validation_data_path = local_validation_data_path
    self.local_dict_path_prefix = local_dict_path_prefix

    self.x = None
    self.y = None
    self.x_valid = None
    self.y_valid = None

  # internal function to Read data file
  def read_data_file(self, filepath):
    # Read data file
    x = []
    y = []
    with open(filepath, 'r', encoding='utf8') as fin:
      for line in fin:
        line = line.strip()
        if len(line) > 0:
          tokens = line.split('\t')
          line_x = tokens[0].strip()
          words_x = line_x.split(' ')
          x.append(words_x)
          words_y = None
          if len(tokens) > 0:
            line_y = tokens[1].strip()
            words_y = line_y.split(' ')
          else:
            # For Auto-Encoder (X=Y)
            words_y = words_x
          y.append(words_y)                
    return (x, y)

  # Get all data as list (probably for debug propose)
  def load_as_list(self):
    # Implement Lazy file reading
    if self.x is None or self.y is None or self.x_valid is None or self.y_valid is None:
      (x, y) = self.read_data_file(self.local_train_data_path)
      (x_valid, y_valid) = self.read_data_file(self.local_validation_data_path)
      self.x = x
      self.y = y
      self.x_valid = x_valid
      self.y_valid = y_valid
    return (self.x, self.y, self.x_valid, self.y_valid)

  # Get unique data from the dataset as list
  def get_unique_data(self, column_id):
    # Implement lazy load/cache dict file
    unique_data = set()
    dict_file_path = self.local_dict_path_prefix + str(column_id) + '.txt'
    if not os.path.exists(dict_file_path):
      (x, y, _, _) = self.load_as_list()
      data = []
      if column_id == 0:
        data = [self.x]
      elif column_id == 1:
        data = [self.y]
      elif column_id == -1: # For combined dict (Used in case of Shared Word Embedding between input and output)
        data = [self.x, self.y]
      for each_data in data:
        for line in each_data:
          for word in line:
            unique_data.add(word)
      with open(dict_file_path, 'w', encoding='utf8') as fout:
        for word in unique_data:
          fout.write(word + '\n')
    else:
      with open(dict_file_path, 'r', encoding='utf8') as fin:
        for line in fin:
          word = line.strip()
          unique_data.add(word)
          
    return sorted(list(unique_data))

  # Get local directory that store data
  def get_local_data_dir(self):
    return self.local_data_dir
