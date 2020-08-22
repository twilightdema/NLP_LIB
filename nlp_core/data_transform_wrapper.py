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

  # Function indicates of the data transform has aggregated transformation applied on raw dataset or not.
  # Example is that BERT pretrained data transform will try to batch many lines of text from dataset.load_as_list()
  # into single data row to maximize length of tranformed dataset.
  # For such case, in model training, we should not use dataset.load_as_list() and call transform.encode one by one row
  # but instead we should load already transformed data. The flag is to indicate which loading approach to be used.
  # Note that encode/decode function should still be implemented because we will call it in online inference mode.
  def is_data_preaggregated(self):
    return False

  # If data is pre-aggregated, this function is called to load pre-aggregated data instead of calling dataset.load_as_list()
  # Returns from this function should be (X, Y, X_valid, Y_valid)
  def load_preaggregated_data(self):
    return None
