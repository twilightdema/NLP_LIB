import urllib.request
import sys, os, re
import zipfile

from NLP_LIB.datasets.gcs_dataset_wrapper import DatasetWrapper

# Wrapper class for CoLA Dataset.
class ColaDatasetWrapper(DatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'cola'

    super(ColaDatasetWrapper, self).__init__(config)

    base_data_dir = config['base_data_dir']
    dataset_name = config['dataset_name']

    source_zip_file_url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
 
    local_data_dir = os.path.join(base_data_dir, dataset_name)
    if not os.path.exists(local_data_dir):
      os.makedirs(local_data_dir)

    local_zip_file_path = os.path.join(local_data_dir, 'original.zip')
    local_train_data_path = os.path.join(local_data_dir, 'cola_public', 'tokenized', 'in_domain_train.tsv')
    local_validation_data_path = os.path.join(local_data_dir, 'cola_public', 'tokenized', 'in_domain_dev.tsv')
    local_dict_path_prefix = os.path.join(local_data_dir, 'dict')

    if not os.path.exists(local_train_data_path) or not os.path.exists(local_validation_data_path):
      print('Downloading from: ' + source_zip_file_url + ' to ' + local_zip_file_path)
      urllib.request.urlretrieve(source_zip_file_url, local_zip_file_path)

      with zipfile.ZipFile(local_zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(local_data_dir)

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
          if len(tokens) >= 3:
            line_x = tokens[3].strip()
            words_x = line_x.split(' ')
            x.append(words_x)
            words_y = None
            line_y = tokens[1].strip()
            words_y = line_y.split(' ')
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

  # Get token seperator (Ex. ' ' for English, '' for Thai - because no seperator)
  def get_trivial_token_separator(self):
    return ' '
    
# Unit Test
if __name__ == '__main__':
  obj = ColaDatasetWrapper({'base_data_dir': 'tmp'})
  x, y, x_valid, y_valid = obj.load_as_list()
  print('X => ' + str(len(x)))
  print('Y => ' + str(len(y)))
  print('X_Valid => ' + str(len(x_valid)))
  print('Y_Valid => ' + str(len(y_valid)))
  print(x[0])
  print(y[0])
