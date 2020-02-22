import numpy as np
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper

class SingleClassTransformWrapper(DataTransformWrapper):
  
  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor
  def __init__(self, config, dataset):
    super(SingleClassTransformWrapper, self).__init__(config, dataset)  
    column_id = config['column_id']
    print('Column ID = ' + str(column_id))
    token_list = dataset.get_unique_data(column_id)
    print('Number of class = ' + str(len(token_list)))
    token_list = sorted(token_list)
    self.id2t = token_list
    self.t2id = {v:k for k,v in enumerate(self.id2t)}

  def id(self, x):	return self.t2id.get(x)
  def token(self, x):	return self.id2t[x]

  # Function used for encode batch of string data into batch of encoded integer
  def encode(self, token_list, max_length = 999):
    X = np.zeros((len(token_list), self.num()), dtype='float32')
    for i, x in enumerate(token_list):
      for each_x in x:
        cls_id = self.t2id.get(each_x)
        X[i, cls_id] = 1.0
    return X
  
  # Function used for decode batch of integers back to batch of string
  def decode(self, id_list):
    ret = []
    for i, x in enumerate(id_list):
      data = self.token(x)
      ret.append(data)
    return ret

  # Function to return size of dictionary (key size)
  def num(self):
    return len(self.id2t)
  