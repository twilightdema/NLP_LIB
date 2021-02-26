import sys
import os
import zipfile
import pickle
import numpy as np
import sentencepiece as spm
import requests

from dataset_util import VOCAB_SIZE, TOKEN_UNKNOWN, TOKEN_CLS, TOKEN_SEP, \
  TOKEN_CLS_STATIC_EMBEDDING, TOKEN_SEP_STATIC_EMBEDDING, \
  balance_training_data, load_word_embedding_model, perform_word_embedding


#################################################################
# FUNCTIONS FOR LOADING COLA DATASET
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

def load_encoded_cola_data_static_word_embedding(perform_data_balance):
  encoded_data_train_path = os.path.join('dataset', 'cola', 'train_static_word_embedding.pk')
  encoded_data_dev_path = os.path.join('dataset', 'cola', 'dev_static_word_embedding.pk')

  data_train = None
  data_dev = None

  if os.path.exists(encoded_data_train_path) and os.path.exists(encoded_data_dev_path):
    # Load from processed file
    print('[INFO] Loading data from pre-generated file.')
    with open(encoded_data_train_path,'rb') as fin:
      data_train = pickle.load(fin)
    with open(encoded_data_dev_path,'rb') as fin:
      data_dev = pickle.load(fin)

    if perform_data_balance:
      print('[INFO] Perform Data Balancing')
      data_train = balance_training_data(data_train)
      data_dev = balance_training_data(data_dev)

    return data_train, data_dev

  data_train_, data_dev_ = load_cola_data()

  # Perform word embedding, we remove all data entry those contain OOV words.
  embedding_model = load_word_embedding_model()
  data_train = []
  data_dev = []
  for data in data_train_:
    encoded_data = perform_word_embedding(data['input'], embedding_model)
    if encoded_data is not None:
      data['input_ids'] = encoded_data
      data_train.append(data)
  for data in data_dev_:
    encoded_data = perform_word_embedding(data['input'], embedding_model)
    if encoded_data is not None:
      data['input_ids'] = encoded_data
      data_dev.append(data)

  print('[INFO] ' + str(len(data_train)) + ' out of ' + str(len(data_train_)) + ' are embeddable (Train)...')
  print('[INFO] ' + str(len(data_dev)) + ' out of ' + str(len(data_dev_)) + ' are embeddable (Dev)...')

  # Save pre-generated file
  with open(encoded_data_train_path, 'wb') as fout:
    pickle.dump(data_train, fout)
  with open(encoded_data_dev_path, 'wb') as fout:
    pickle.dump(data_dev, fout)

  if perform_data_balance:
    print('[INFO] Perform Data Balancing')
    data_train = balance_training_data(data_train)
    data_dev = balance_training_data(data_dev)

  return data_train, data_dev

def load_encoded_cola_data_spm(perform_data_balance):
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

    if perform_data_balance:
      print('[INFO] Perform Data Balancing')
      data_train = balance_training_data(data_train)
      data_dev = balance_training_data(data_dev)

    return data_train, data_dev

  data_train, data_dev = load_cola_data()
  max_dict_size = VOCAB_SIZE
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
    spm.SentencePieceTrainer.Train('--pad_id=0 --bos_id=' + str(TOKEN_CLS) + 
      ' --eos_id=' + str(TOKEN_SEP) + 
      ' --unk_id=' + str(TOKEN_UNKNOWN) + 
      ' --user_defined_symbols=<MASK> --input=' + 
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

  if perform_data_balance:
    print('[INFO] Perform Data Balancing')
    data_train = balance_training_data(data_train)
    data_dev = balance_training_data(data_dev)

  return data_train, data_dev
