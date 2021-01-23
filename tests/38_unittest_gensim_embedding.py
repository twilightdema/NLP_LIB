import numpy as np
import gensim.downloader as api
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import Word

import os
import sys
import requests
import zipfile
import pickle
import math
import random
import json

# Flag choosing if we want to balance Training / Test data for fairness
PERFORM_DATA_BALANCING = True

#################################################################################################
# Function for performing static word embedding
def load_word_embedding_model():
  # Init (and download) necessary model and data
  nltk.download('stopwords')
  nltk.download('wordnet')
  nltk.download('punkt')
  model = api.load("glove-wiki-gigaword-100")  # download the model and return as object ready for use
  return model

def perform_word_embedding(sentence, embedding_model):
  stop = stopwords.words('english')
  x = ' '.join(word_tokenize(sentence))
  x = ' '.join(x.lower() for x in x.split())
  # x = ' '.join(x for x in x.split() if x not in string.punctuation)
  x = x.replace('[^\w\s]','')
  print('1: ' + x)
  # x = ' '.join(x for x in x.split() if not x.isdigit())
  print('2: ' + x)
  # x = ' '.join(x for x in x.split() if not x in stop)
  print('3: ' + x)
  x = ' '.join([Word(word).lemmatize() for word in x.split()])
  print('4: ' + x)
  x = x.split()
  try:
    return np.array([embedding_model[w] for w in x])
  except Exception as e:
    print('Exception during perform embedding: ' + str(e))
    return None

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

def balance_training_data(data):
  label_count_map = {}
  label_to_data_map = {}
  for entry in data:
    label = entry['label']
    if label in label_count_map:
      label_count_map[label] = label_count_map[label] + 1
      label_to_data_map[label].append(entry)
    else:
      label_count_map[label] = 1
      label_to_data_map[label] = [entry]
  labels = list(label_count_map.keys())
  balanced_data = []
  while True:
    selected_label = labels[random.randint(0, len(labels)-1)]      
    if len(label_to_data_map[selected_label]) == 0:        
      break
    entry = label_to_data_map[selected_label].pop()
    balanced_data.append(entry)
  return balanced_data

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

def load_encoded_cola_data_static_word_embedding():
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

    if PERFORM_DATA_BALANCING:
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

  if PERFORM_DATA_BALANCING:
    print('[INFO] Perform Data Balancing')
    data_train = balance_training_data(data_train)
    data_dev = balance_training_data(data_dev)

  return data_train, data_dev

print('Unit testing, load CoLA data using static word embedding...')
data_train, data_dev = load_encoded_cola_data_static_word_embedding()
print('Sample training entry:')
print(data_train[0])
print('Train size: ' + str(len(data_train)))
print('Dev size: ' + str(len(data_dev)))
