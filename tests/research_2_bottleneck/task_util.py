import numpy as np
from dataset_util import VOCAB_SIZE, TOKEN_UNKNOWN, TOKEN_CLS, TOKEN_SEP
from task_cola import load_encoded_cola_data_spm, load_encoded_cola_data_static_word_embedding
from task_mrpc import load_encoded_mrpc_data_spm, load_encoded_mrpc_data_static_word_embedding

# Function to load dataset given task name
def get_dataset_loader_func(task_name):
  if task_name == 'cola':
    return load_encoded_cola_data_spm, load_encoded_cola_data_static_word_embedding
  elif task_name == 'mrpc':
    return load_encoded_mrpc_data_spm, load_encoded_mrpc_data_static_word_embedding
  else:
    return None, None

def truncate_and_pad(ar, target_len, trainable_embedding_layer, d_model):
  # Add [CLS] and [SEP] token infront and after input sequence respectively
  # TODO: Is this valid??
  # In case of static word embedding we use [CLS] as "all 1.0" 
  #  and [SEP] as "all -1.0" embedding vector
  #  and [PAD] as "all 0.0" embedding vector.
  if trainable_embedding_layer:
    target_len = target_len + 0 # We already factor +2 special tokens
    ret = []
    mask = []
    ret.append(TOKEN_CLS)
    mask.append(0.0)
    for tok in ar:
      ret.append(tok)
      mask.append(0.0)
    ret.append(TOKEN_SEP)
    mask.append(0.0)
    ret = ret[0:target_len]
    mask = mask[0:target_len]
    while len(ret) < target_len:
      ret.append(0)
      mask.append(1.0)
    return ret, mask
  else:
    target_len = target_len + 0 # We already factor +2 special tokens
    embedding_dimension = d_model
    ret = []
    mask = []
    ret.append(np.array([trainable_embedding_layer] * embedding_dimension)) # TOKEN_CLS
    mask.append(0.0)
    for tok in ar:
      ret.append(tok)
      mask.append(0.0)
    ret.append(np.array([trainable_embedding_layer] * embedding_dimension)) # TOKEN_SEP
    mask.append(0.0)
    ret = ret[0:target_len]
    mask = mask[0:target_len]
    while len(ret) < target_len:
      ret.append(np.array([0.0] * embedding_dimension)) # TOKEN_PAD
      mask.append(1.0)
    return ret, mask

# Function to generate generate input_id, label, mask from dataset
def dataset_to_features(seq_len, dataset, trainable_embedding_layer, d_model):
  input_ids_batch = [a['input_ids'] for a in dataset]
  input_batch = []
  mask_batch = []
  for input_ids in input_ids_batch:
    ids, masks = truncate_and_pad(input_ids, seq_len, trainable_embedding_layer, d_model)
    input_batch.append(ids)
    mask_batch.append(masks)
  
  # Transform label into one-hot multi-class classification format
  label_batch = [[0.0, 0.0] for _ in dataset]
  for label_ele, a in zip(label_batch, dataset):
    label_ele[int(a['label'])] = 1.0

  # In TF2 implementation, we want mask with dimension of (BATCH, 1, 1, SEQ_LEN)
  mask_batch_np = np.array(mask_batch)
  mask_batch_np = np.expand_dims(mask_batch_np, (1,2))

  return np.array(input_batch), mask_batch_np, np.array(label_batch)
