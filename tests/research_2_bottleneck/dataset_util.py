import gensim.downloader as api
import random
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import Word

VOCAB_SIZE = 150

# String speical token specifications for Sentencepiece model
TOKEN_UNKNOWN = 1
TOKEN_CLS = 2
TOKEN_SEP = 3

# String speical token specifications for Static Embedding
TOKEN_CLS_STATIC_EMBEDDING = 1.0
TOKEN_SEP_STATIC_EMBEDDING = -1.0

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
  x = x.replace('[^\w\s]','')
  x = ' '.join([Word(word).lemmatize() for word in x.split()])
  x = x.split()
  try:
    return np.array([embedding_model[w] for w in x])
  except Exception as e:
    print('Exception during perform embedding: ' + str(e))
    return None

# Function to perform data balancing as per label
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

  for label in labels:
    print('Label: ' + str(label) + ' has ' + str(label_count_map[label]) + ' rows')

  balanced_data = []
  while True:
    selected_label = labels[random.randint(0, len(labels)-1)]      
    if len(label_to_data_map[selected_label]) == 0:        
      break
    entry = label_to_data_map[selected_label].pop()
    balanced_data.append(entry)

  label_count_map = {}
  label_to_data_map = {}
  for entry in balanced_data:
    label = entry['label']
    if label in label_count_map:
      label_count_map[label] = label_count_map[label] + 1
      label_to_data_map[label].append(entry)
    else:
      label_count_map[label] = 1
      label_to_data_map[label] = [entry]
  labels = list(label_count_map.keys())

  for label in labels:
    print('Balanced Label: ' + str(label) + ' has ' + str(label_count_map[label]) + ' rows')

  return balanced_data
