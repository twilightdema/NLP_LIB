import numpy as np
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper

# The class for transforming batch data into single last token. If token count is shorter than limit, it will use endid() for prediction target
class LastTokenExtractionWrapper(DataTransformWrapper):
  
  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor
  def __init__(self, config, dataset):
    super(LastTokenExtractionWrapper, self).__init__(config, dataset)  
    column_id = config['column_id']
    print('Column ID = ' + str(column_id))
    token_list = dataset.get_unique_data(column_id)
    #print('Token List = ')
    #print(token_list)
    self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>'] + token_list
    self.t2id = {v:k for k,v in enumerate(self.id2t)}

  def id(self, x):	return self.t2id.get(x, 1)
  def token(self, x):	return self.id2t[x]
  def startid(self):  return 2
  def endid(self):    return 3

  # Function used for encode batch of string data into batch of encoded integer
  def encode(self, token_list, max_length = 999):
    X = np.zeros((len(token_list), 1), dtype='int32')
    for i, x in enumerate(token_list):
      x = x[:max_length-1]
      if 1 + len(x) < max_length:
        X[i, 0] = self.endid()
      else:
        X[i, 0] = self.id(x[-1])
    return X
  
  # Function used for decode batch of integers back to batch of string
  def decode(self, id_list):
    ret = []
    for i, x in enumerate(id_list):
      text = ''
      for j, z in enumerate(x):
        text = text + self.token(z)
      ret.append(text)
    return ret

  # Function to return size of dictionary (key size)
  def num(self):
    return len(self.id2t)
  
  # Function to return list of objects to differentiate cached of input/output that model will use.
  # Basically it is configurations that effect encoded data.
  def get_data_effected_configs(self):
    return '_'
