# Base class for dataset
class DatasetWrapper:

  # When initialize Dataset, we pass configuration to constructor
  def __init__(self, config):
    self.config = config

  # Get all data as list (probably for debug propose)
  def load_as_list(self):
    return []
    
  # Get unique data from the dataset as list
  def get_unique_data(self, column_id):
    return []

  # Get local directory that store data
  def get_local_data_dir(self):
    return None

  # Get token seperator (Ex. ' ' for English, '' for Thai - because no seperator)
  def get_trivial_token_separator(self):
    return ''

  # Perform post-processing on fully loaded data (Maybe filter or some custom logic on dataset setting)
  def postprocess_data_loading(self, X, Y, X_Valid, Y_Valid, data_transform, column_id):
    return X, Y, X_Valid, Y_Valid
    