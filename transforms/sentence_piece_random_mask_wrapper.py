import numpy as np
import os, sys
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper
import sentencepiece as spm
import random

class SentencePieceRandomMaskWrapper(DataTransformWrapper):
  
  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor
  def __init__(self, config, dataset):
    super(SentencePieceRandomMaskWrapper, self).__init__(config, dataset)  
    column_id = config['column_id']
    self.percent_mask = config['percent_mask']
    self.percent_mask_correct = config['percent_mask_correct']
    self.percent_mask_incorrect = config['percent_mask_incorrect']
    min_freq = 0
    max_dict_size = 15000
    if 'max_dict_size' in config and config['max_dict_size'] is not None:
      max_dict_size = config['max_dict_size']
    self.max_dict_size = max_dict_size
    self.sentence_piece_processor = spm.SentencePieceProcessor()
    self.trivial_token_separator = dataset.get_trivial_token_separator()

    print('Max Dictionary Size = ' + str(max_dict_size))

    print('Column ID = ' + str(column_id))

    # Load from dict from cache if possible
    local_data_dir = dataset.get_local_data_dir()
    local_dict_path_prefix = os.path.join(local_data_dir, 'dict_' + 
      type(self).__name__ + 
      '_dict' + str(max_dict_size))

    local_dict_vocab_path = local_dict_path_prefix + str(column_id) + '.vocab'
    local_dict_model_path = local_dict_path_prefix + str(column_id) + '.model'
    local_untokened_data_file = local_dict_path_prefix + str(column_id) + '.untoken'

    if not os.path.exists(local_dict_model_path):

      # Create untokened data file
      with open(local_untokened_data_file, 'w', encoding='utf-8') as fout:
        print('Constructing untokened document')
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
            untokened_line = ''
            for word in line:
              if len(untokened_line) > 0:
                untokened_line = untokened_line + self.trivial_token_separator              
              untokened_line = untokened_line + word
            fout.write(untokened_line + '\n')
        

      # Train sentence piece model
      spm.SentencePieceTrainer.Train('--pad_id=0 --bos_id=2 --eos_id=3 --unk_id=1 --user_defined_symbols=<MASK> --input=' + 
        local_untokened_data_file + 
        ' --model_prefix=sp --vocab_size=' + str(max_dict_size))

      # Delete untokened data file
      os.remove(local_untokened_data_file)

      # Move sp.model / sp.vocab to the dict paths
      os.rename("sp.model", local_dict_model_path)
      os.rename("sp.vocab", local_dict_vocab_path)

      self.sentence_piece_processor.Load(local_dict_model_path)
                  
    else:
      self.sentence_piece_processor.Load(local_dict_model_path)

    print('Dictionary size = ' +str(self.sentence_piece_processor.GetPieceSize()))

    # For reproducible of data
    random.seed(0)

  def startid(self):  return 2
  def endid(self):    return 3
  def maskid(self):    return 4

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
      x = x[:max_length - 1]
      x = self.trivial_token_separator.join(x).strip()
      encoded_x = self.sentence_piece_processor.EncodeAsIds(x)
      #sys.stdout.buffer.write(x.encode('utf8') + '\n'.encode('utf8'))
      # Ensure that we are not 
      encoded_x = encoded_x[:max_length - 1]
      X[i, 1:len(encoded_x) + 1] = encoded_x

      # If sentence is not end, then don't add end symbol at the end of encoded tokens
      # We have to mask out last token in some case (Language Model). Note that masked token can be endid() (predict end of sequence)
      if 1 + len(encoded_x) < max_length:
        X[i,1 + len(encoded_x)] = self.endid()

      # If clf_pos_offset is specified, we trim data to the length and set clf_id at the position
      if clf_pos_offset is not None:
        clf_pos = min(1 + len(encoded_x), max_length - 1 + clf_pos_offset)
        X[i, clf_pos] = clf_id
        X[i, clf_pos + 1:] = 0

      # Mask some of data
      num_to_mask = int(self.percent_mask * len(encoded_x) / 100)
      #print('self.percent_mask = ' + str(self.percent_mask) + ', num_to_mask = ' + str(num_to_mask))
      if num_to_mask == 0 and len(encoded_x) > 1 and self.percent_mask > 0:
        num_to_mask = 1
      count = 0
      #print('len(encoded_x) = ' + str(len(encoded_x)) + ', num_to_mask## = ' + str(num_to_mask))
      while count < num_to_mask:
        pos = random.randint(1, len(encoded_x))
        if X[i, pos] != self.maskid():
          mask_choice = random.randint(0, 100)
          if mask_choice < self.percent_mask_correct:
            pass
          elif mask_choice < self.percent_mask_correct + self.percent_mask_incorrect:
            X[i, pos] = random.randint(self.maskid() + 1, self.num() - 1)
          else:
            X[i, pos] = self.maskid()
          count = count + 1

      #print('Encoded Ids = ' + str(X[i,:]))

    return X
  
  # Function used for decode batch of integers back to batch of string
  def decode(self, id_list):
    ret = []
    for i, x in enumerate(id_list):
      text = self.sentence_piece_processor.DecodeIds(x)
      ret.append(text)
    return ret

  # Function to return size of dictionary (key size)
  def num(self):
    return self.sentence_piece_processor.GetPieceSize()
  
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

    return '_dict' + str(self.max_dict_size) + '_mask' + str(self.percent_mask) + '_' + clf_txt
