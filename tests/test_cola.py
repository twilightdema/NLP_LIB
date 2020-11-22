# This unit test case is for tesing CoLA dataset load and processing
import os
import sys
import requests
import zipfile
import pickle
import numpy as np
import sentencepiece as spm

def check_and_download_cola():
  data_folder = os.path.join('dataset', 'cola')
  if not os.path.exists(data_folder):
    print('[INFO] No data folder found, recreating...')
    os.makedirs(data_folder)
  zip_filename = 'cola_public_1.1.zip'
  zip_filepath = os.path.join(data_folder, zip_filename)
  if not os.path.exists(zip_filepath):
    print('[INFO] No zip file found, downloading...')
    url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
    r = requests.get(url, allow_redirects=True)
    open(zip_filepath, 'wb').write(r.content)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
      zip_ref.extractall(data_folder)

def read_cola_data_file(file_path):
  data = []
  with open(file_path, 'r', encoding='utf-8') as fin:
    for line in fin:
      columns = line.split('\t')
      data_row = {
        'id': columns[0].strip(),
        'label': columns[1].strip(),
        'input': columns[3].strip()
      }
      print(data_row['input'] + ' => ' + data_row['label'])
      data.append(data_row)
  return data

def load_cola_data():
  check_and_download_cola()
  data_path_train = os.path.join('dataset', 'cola', 'cola_public', 'raw', 'in_domain_train.tsv')
  data_path_dev = os.path.join('dataset', 'cola', 'cola_public', 'raw', 'in_domain_dev.tsv')
  data_train = read_cola_data_file(data_path_train)
  data_dev = read_cola_data_file(data_path_dev)
  return data_train, data_dev

def load_encoded_cola_data():
  encoded_data_train_path = os.path.join('dataset', 'cola', 'train.pk')
  encoded_data_dev_path = os.path.join('dataset', 'cola', 'dev.pk')

  data_train = None
  data_dev = None

  if os.path.exists(encoded_data_train_path) and os.path.exists(encoded_data_dev_path):
    # Load from processed file
    print('[INFO] Loading data from pre-generated file.')
    with open(encoded_data_train_path,'rb') as fin:
      data_train = pickle.load(fin)
    with open(encoded_data_dev_path,'rb') as fin:
      data_dev = pickle.load(fin)

    return data_train, data_dev

  data_train, data_dev = load_cola_data()
  max_dict_size = 150
  sentence_piece_processor = spm.SentencePieceProcessor()
  print('[INFO] Max Dictionary Size = ' + str(max_dict_size))
  dict_vocab_path = os.path.join('dataset', 'cola', 'spm.vocab')
  dict_model_path = os.path.join('dataset', 'cola', 'spm.model')

  if not os.path.exists(dict_model_path):
    print('[INFO] No SPM model file, creating...')
    # Create raw corpus file to train SPM
    raw_corpus_file = os.path.join('dataset', 'cola', 'corpus.txt')
    with open(raw_corpus_file, 'w', encoding='utf-8') as fout:
      for data_row in data_train:
        fout.write(data_row['input'] + '\n')
      
    # Train sentence piece model
    spm.SentencePieceTrainer.Train('--pad_id=0 --bos_id=2 --eos_id=3 --unk_id=1 --user_defined_symbols=<MASK> --input=' + 
      raw_corpus_file + 
      ' --model_prefix=sp --vocab_size=' + str(max_dict_size) + ' --hard_vocab_limit=false')

    # Delete raw corpus file
    os.remove(raw_corpus_file)

    # Move sp.model / sp.vocab to the dict paths
    os.rename("sp.model", dict_model_path)
    os.rename("sp.vocab", dict_vocab_path)

    sentence_piece_processor.Load(dict_model_path)            
  else:
    sentence_piece_processor.Load(dict_model_path)

  print('[INFO] Dictionary size = ' +str(sentence_piece_processor.GetPieceSize()))

  # Perform encoding
  for data in data_train:
    encoded_data = sentence_piece_processor.EncodeAsIds(data['input'])
    data['input_ids'] = encoded_data
  for data in data_dev:
    encoded_data = sentence_piece_processor.EncodeAsIds(data['input'])
    data['input_ids'] = encoded_data

  # Save pre-generated file
  with open(encoded_data_train_path, 'wb') as fout:
    pickle.dump(data_train, fout)
  with open(encoded_data_dev_path, 'wb') as fout:
    pickle.dump(data_dev, fout)

  return data_train, data_dev

data_train, data_dev = load_encoded_cola_data()
print('Training data has ' + str(len(data_train)) + ' rows.')
print('Validation data has ' + str(len(data_dev)) + ' rows.')
