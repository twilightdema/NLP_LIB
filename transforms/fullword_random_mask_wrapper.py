import numpy as np
import random
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper

class FullWordRandomMaskWrapper(DataTransformWrapper):
  
  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor
  def __init__(self, config, dataset):
    super(FullWordRandomMaskWrapper, self).__init__(config, dataset)  
    column_id = config['column_id']
    self.percent_mask = config['percent_mask']
    self.percent_mask_correct = config['percent_mask_correct']
    self.percent_mask_incorrect = config['percent_mask_incorrect']
    print('Column ID = ' + str(column_id))
    token_list = dataset.get_unique_data(column_id)
    #print('Token List = ')
    #print(token_list)
    self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>', '<MASK>'] + token_list
    self.t2id = {v:k for k,v in enumerate(self.id2t)}
    self.trivial_token_separator = dataset.get_trivial_token_separator()

    # For reproducible of data
    random.seed(0)

  def id(self, x):	return self.t2id.get(x, 1)
  def token(self, x):	return self.id2t[x]
  def startid(self):  return 2
  def endid(self):    return 3
  def maskid(self):   return 4

  # Function used for encode batch of string data into batch of encoded integer
  def encode(self, token_list, max_length = 999):

    # This is to force placing special clf_id not exceed specific location (Such as len-1 in decoder only architecture because it trims the last token out)
    clf_id = None
    clf_pos_offset = None
    if 'clf_id' in self.config:
      clf_id = self.config['clf_id']
    if 'clf_pos_offset' in self.config:
      clf_pos_offset = self.config['clf_pos_offset']

    X = np.zeros((len(token_list), max_length), dtype='int32')
    X[:,0] = self.startid()    
    for i, x in enumerate(token_list):
      x = x[:max_length-1]

      for j, z in enumerate(x):
        X[i,1+j] = self.id(z)
      # If sentence is not end, then don't add end symbol at the end of encoded tokens
      # We have to mask out last token in some case (Language Model). Note that masked token can be endid() (predict end of sequence)
      if 1 + len(x) < max_length:
        X[i,1 + len(x)] = self.endid()

      # If clf_pos_offset is specified, we trim data to the length and set clf_id at the position
      if clf_pos_offset is not None:
        clf_pos = min(1 + len(x), max_length - 1 + clf_pos_offset)
        X[i, clf_pos] = clf_id
        X[i, clf_pos + 1:] = 0

      # Mask some of data
      num_to_mask = int(self.percent_mask * len(x) / 100)
      if num_to_mask == 0 and len(x) > 1 and self.percent_mask > 0:
        num_to_mask = 1
      count = 0
      while count < num_to_mask:
        pos = random.randint(1, len(x))
        if X[i, pos] != self.maskid():
          mask_choice = random.randint(0, 100)
          if mask_choice < self.percent_mask_correct:
            pass
          elif mask_choice < self.percent_mask_correct + self.percent_mask_incorrect:
            X[i, pos] = random.randint(self.maskid() + 1, self.num() - 1)
          else:
            X[i, pos] = self.maskid()
          count = count + 1

    return X
  
  # Function used for decode batch of integers back to batch of string
  def decode(self, id_list):
    ret = []
    for i, x in enumerate(id_list):
      text = ''
      for j, z in enumerate(x):
        if len(text) > 0:
          text = text + self.trivial_token_separator
        text = text + self.token(z)
      ret.append(text)
    return ret

  # Function to return size of dictionary (key size)
  def num(self):
    return len(self.id2t)
  
  # Function to return list of objects to differentiate cached of input/output that model will use.
  # Basically it is configurations that effect encoded data.
  def get_data_effected_configs(self):
    clf_id = None
    clf_pos_offset = None
    if 'clf_id' in self.config:
      clf_id = self.config['clf_id']
    if 'clf_pos_offset' in self.config:
      clf_pos_offset = self.config['clf_pos_offset']
    clf_txt = ''
    if clf_pos_offset is not None:
      clf_txt = '_clf' + str(clf_id) + 'at' + str(clf_pos_offset)

    return '_mask' + str(self.percent_mask) + '_' + clf_txt
