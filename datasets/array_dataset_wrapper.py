from NLP_LIB.nlp_core.dataset_wrapper import DatasetWrapper

# Subclass of DatasetWrapper that wrap generic python array
class ArrayDatasetWrapper(DatasetWrapper):

  # Initialize from generic python array
  def __init__(self, config):
    super(ArrayDatasetWrapper, self).__init__(config)  
    (X, Y, X_valid, Y_valid) = config['values']
    self.x = X
    self.y = Y
    self.x_valid = X_valid
    self.y_valid = Y_valid

  # Get all data as list (probably for debug propose)
  def load_as_list(self):
    return (self.x, self.y, self.x_valid, self.y_valid)

  # Get unique data from the dataset as list
  def get_unique_data(self, column_id):
    unique_data = set()
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
    return sorted(list(unique_data))

  # Get local directory that store data
  def get_local_data_dir(self):
    return None
