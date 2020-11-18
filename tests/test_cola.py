# This unit test case is for tesing CoLA dataset load and processing
import os
import sys
import requests
import zipfile
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
  data_train, data_dev = load_cola_data()
  

data_train, data_dev = load_cola_data()
