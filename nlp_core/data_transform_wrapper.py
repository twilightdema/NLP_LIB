class DataTransformWrapper:

  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor
  def __init__(self, config, dataset):
    self.config = config
    self.dataset = dataset

  # Function used for encode batch of string data into batch of encoded integer
  def encode(self, token_list, max_length = 999):
    return []
  
  # Function used for decode batch of integers back to batch of string
  def decode(self, id_list):
    return ""
  
  # Function to return size of dictionary (key size)
  def num(self):
    return 0

  # Function to return list of objects to differentiate cached of input/output that model will use.
  # Basically it is configurations that effect encoded data.
  def get_data_effected_configs(self):
    return '_'
