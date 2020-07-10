import numpy as np
import os
import random
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper

class BigramRandomMaskWrapper(DataTransformWrapper):
  
  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor
  def __init__(self, config, dataset):
    super(BigramRandomMaskWrapper, self).__init__(config, dataset)  
    column_id = config['column_id']
    self.percent_mask = config['percent_mask']
    self.percent_mask_correct = config['percent_mask_correct']
    self.percent_mask_incorrect = config['percent_mask_incorrect']    
    min_freq = 0
    max_dict_size = 0
    if 'min_freq' in config and config['min_freq'] is not None:
      min_freq = config['min_freq']
    if 'max_dict_size' in config and config['max_dict_size'] is not None:
      max_dict_size = config['max_dict_size']
    self.min_freq = min_freq
    self.max_dict_size = max_dict_size
    self.trivial_token_separator = dataset.get_trivial_token_separator()

    print('Max Dictionary Size = ' + str(max_dict_size))

    print('Column ID = ' + str(column_id))

    # Load from dict from cache if possible
    local_data_dir = dataset.get_local_data_dir()
    local_dict_path_prefix = os.path.join(local_data_dir, 'dict_' + 
      type(self).__name__ + 
      '_dict' + str(max_dict_size) +
      '_min' + str(min_freq) + '_')

    local_dict_path = local_dict_path_prefix + str(column_id) + '.txt'

    unique_data = set()
    if not os.path.exists(local_dict_path):
      
      valid_dict_size = False
      while not valid_dict_size:
        gram_to_count = {}
        print('Constructing Bi-Gram dictionary')
        (x, y, _, _) = dataset.load_as_list()
        data = []
        if column_id == 0:
          data = [x]
        elif column_id == 1:
          data = [y]
        elif column_id == -1:
          data = [x, y]
        
        for each_data in data:
          for line in each_data:
            grams = []
            for word in line:
              grams.append(word)
              if len(grams) == 2:
                grams_string = '|'.join(grams)
                # Ensure that we create dict for only grams those meet minimum occurrence.
                if grams_string in gram_to_count:
                  gram_to_count[grams_string] = gram_to_count[grams_string] + 1
                else:
                  gram_to_count[grams_string] = 1
                if gram_to_count[grams_string] >= min_freq:            
                  unique_data.add(grams_string)
                grams = grams[1:]
        
        dict_size = len(unique_data)
        print('With min freq = ' + str(min_freq) + ', dict_size = ' + str(dict_size))
        if max_dict_size == 0:
          valid_dict_size = True
        else:
          if dict_size <= max_dict_size:
            valid_dict_size = True
          else:
            if min_freq == 0:
              min_freq = 1
            else:
              min_freq = min_freq + 5
              unique_data = set()
              print('Try larger min_freq: ' + str(min_freq))
                  
      with open(local_dict_path, 'w', encoding='utf8') as fout:
        for word in unique_data:
          fout.write(word + '\n')

    else:
      with open(local_dict_path, 'r', encoding='utf8') as fin:
        for line in fin:
          word = line.strip()
          unique_data.add(word)

    token_list = sorted(list(unique_data))
    print('Dictionary size = ' +str(len(token_list)))

    # For reproducible of data
    random.seed(0)
    
    #print('Token List = ')
    #print(token_list)
    self.id2t = ['<PAD>', '<UNK>', '<S>', '</S>', '<MASK>'] + token_list
    self.t2id = {v:k for k,v in enumerate(self.id2t)}

  def id(self, x):	return self.t2id.get(x, 1)
  def token(self, x):	return self.id2t[x]
  def startid(self):  return 2
  def endid(self):    return 3
  def maskid(self):   return 4

  # Function used for encode batch of string data into batch of encoded integer
  def encode(self, token_list, max_length = 999):

    # This is
    #  to force placing special clf_id not exceed specific location (Such as len-1 in decoder only architecture because it trims the last token out)
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

      for j in range(len(x) - 1):
        grams = x[j] + '|' + x[j + 1]
        X[i,1+j] = self.id(grams)
      # If sentence is not end, then don't add end symbol at the end of encoded tokens
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
        text = text + '|' + self.token(z)
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

    return '_dict' + str(self.max_dict_size) + '_min' + str(self.min_freq) + '_mask' + str(self.percent_mask) + '_' + clf_txt
