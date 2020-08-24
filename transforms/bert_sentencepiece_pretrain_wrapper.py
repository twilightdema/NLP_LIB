import sys
sys.path.append('.')

import numpy as np
import os
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper
import sentencepiece as spm
import random
import tensorflow as tf
import six
from tensorflow.keras import backend as K
from tensorflow.keras.layers import *

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

class BERTSPMExampleBuilder(object):
  """Given a stream of input text, creates pretraining examples."""

  def __init__(self, spm_model, cls_id, sep_id, mask_id, max_length):
    self._spm_model = spm_model
    self._current_sentences = []
    self._current_length = 0
    self._max_length = max_length
    self._target_length = max_length
    self.cls_id = cls_id
    self.sep_id = sep_id
    self.mask_id = mask_id

  def add_line(self, line):
    print('Add Line: ' + str(line))
    """Adds a line of text to the current example being built."""
    line = line.strip().replace("\n", " ")
    if (not line) and self._current_length != 0:  # empty lines separate docs
      return self._create_example()
    bert_tokids = self._spm_model.EncodeAsIds(line)      
    #bert_tokens = self._tokenizer.tokenize(line)
    #bert_tokids = self._tokenizer.convert_tokens_to_ids(bert_tokens)
    self._current_sentences.append(bert_tokids)
    self._current_length += len(bert_tokids)
    if self._current_length >= self._target_length:
      return self._create_example()
    return None

  def _create_example(self):
    """Creates a pre-training example from the current list of sentences."""
    # small chance to only have one segment as in classification tasks
    # Because we have randomness here, we cannot separate file for input/output column
    # because it can create different data file for X and Y here!!.
    # To keep it in sync, BERT need column_id "0" for both input and output side,
    # But we will use "is_input" field in config to diffrentiate logic in transformation instead!
    if random.random() < 0.1:
      first_segment_target_length = 100000
    else:
      # -3 due to not yet having [CLS]/[SEP] tokens in the input text
      first_segment_target_length = (self._target_length - 3) // 2

    first_segment = []
    second_segment = []
    for sentence in self._current_sentences:
      # the sentence goes to the first segment if (1) the first segment is
      # empty, (2) the sentence doesn't put the first segment over length or
      # (3) 50% of the time when it does put the first segment over length
      if (len(first_segment) == 0 or
          len(first_segment) + len(sentence) < first_segment_target_length or
          (len(second_segment) == 0 and
           len(first_segment) < first_segment_target_length and
           random.random() < 0.5)):
        first_segment += sentence
      else:
        second_segment += sentence

    # trim to max_length while accounting for not-yet-added [CLS]/[SEP] tokens
    first_segment = first_segment[:self._max_length - 2]
    second_segment = second_segment[:max(0, self._max_length -
                                         len(first_segment) - 3)]

    # prepare to start building the next example
    self._current_sentences = []
    self._current_length = 0
    # small chance for random-length instead of max_length-length example
    if random.random() < 0.05:
      self._target_length = random.randint(5, self._max_length)
    else:
      self._target_length = self._max_length

    return self._make_dict_example(first_segment, second_segment)

  def _make_dict_example(self, first_segment, second_segment):
    """Converts two "segments" of text into a tf.train.Example."""
    input_ids = [self.cls_id] + first_segment + [self.sep_id]
    segment_ids = [0] * len(input_ids)
    if second_segment:
      input_ids += second_segment + [self.sep_id]
      segment_ids += [1] * (len(second_segment) + 1)
    input_mask = [1] * len(input_ids)
    input_ids += [0] * (self._max_length - len(input_ids))
    input_mask += [0] * (self._max_length - len(input_mask))
    segment_ids += [0] * (self._max_length - len(segment_ids))
    '''
    dict_example = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "segment_ids": segment_ids
    }
    return dict_example
    '''
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "input_ids": create_int_feature(input_ids),
        "input_mask": create_int_feature(input_mask),
        "segment_ids": create_int_feature(segment_ids)
    }))
    return tf_example

'''
class BERTFullDictExampleWriter(object):
  """Writes pre-training examples to disk."""

  def __init__(self, job_id, vocab_file, output_dir, max_seq_length,
               num_jobs, blanks_separate_docs, do_lower_case,
               num_out_files=1000):
    self._blanks_separate_docs = blanks_separate_docs
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file,
        do_lower_case=do_lower_case)
    self._example_builder = BERTFullDictExampleBuilder(tokenizer, max_seq_length)
    self._writers = []
    for i in range(num_out_files):
      if i % num_jobs == job_id:
        output_fname = os.path.join(
            output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(
                i, num_out_files))
        self._writers.append(tf.io.TFRecordWriter(output_fname))
    self.n_written = 0

  def write_examples(self, input_file):
    """Writes out examples from the provided input file."""
    with tf.io.gfile.GFile(input_file) as f:
      for line in f:
        line = line.strip()
        if line or self._blanks_separate_docs:
          example = self._example_builder.add_line(line)
          if example:
            self._writers[self.n_written % len(self._writers)].write(
                example.SerializeToString())
            self.n_written += 1
      example = self._example_builder.add_line("")
      if example:
        self._writers[self.n_written % len(self._writers)].write(
            example.SerializeToString())
        self.n_written += 1

  def finish(self):
    for writer in self._writers:
      writer.close()
'''

class BERTSPMExampleWriter(object):
  """Writes pre-training examples to disk."""

  def __init__(self, job_id, spm_model, output_dir, max_seq_length,
               num_jobs, blanks_separate_docs, do_lower_case,
               cls_id, sep_id, mask_id,
               num_out_files=1):
    self._blanks_separate_docs = blanks_separate_docs
    self._example_builder = BERTSPMExampleBuilder(spm_model, cls_id, sep_id, mask_id, max_seq_length)
    self._writers = []
    os.makedirs(output_dir)
    for i in range(num_out_files):
      if i % num_jobs == job_id:
        output_fname = os.path.join(
            output_dir, "pretrain_data.tfrecord-{:}-of-{:}".format(
                i, num_out_files))
        self._writers.append(tf.io.TFRecordWriter(output_fname))
    self.n_written = 0

  def write_examples(self, input_file):
    print('*** write_examples: ' + input_file)
    """Writes out examples from the provided input file."""
    with tf.io.gfile.GFile(input_file) as f:
      for line in f:
        line = line.strip()
        print('line = ' + str(line))
        if line or self._blanks_separate_docs:
          example = self._example_builder.add_line(line)
          if example:
            self._writers[self.n_written % len(self._writers)].write(
                example.SerializeToString())
            self.n_written += 1
      example = self._example_builder.add_line("")
      if example:
        self._writers[self.n_written % len(self._writers)].write(
            example.SerializeToString())
        self.n_written += 1

  def finish(self):
    print('>>>>>>>>>>>>>>>>> n_written = ' + str(self.n_written))
    for writer in self._writers:
      writer.close()

class BERTSentencePiecePretrainWrapper(DataTransformWrapper):
  
  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor.
  # For BERT pretrained dataset, we perform encoding at initialization step
  # because we need to separate sentence as chunk 1, 2 and add segment id information.
  # The encode function is used mainly in inference step only as all text will be in segment 0 only.
  def __init__(self, config, dataset):
    super(BERTSentencePiecePretrainWrapper, self).__init__(config, dataset)  
    print('dataset = ' + str(dataset))
    column_id = config['column_id']
    min_freq = 0
    max_dict_size = 15000
    if 'max_dict_size' in config and config['max_dict_size'] is not None:
      max_dict_size = config['max_dict_size']
    self.max_dict_size = max_dict_size
    self.sentence_piece_processor = spm.SentencePieceProcessor()
    self.trivial_token_separator = dataset.get_trivial_token_separator()
    self.max_seq_length =  config['max_seq_length']
    self.preaggregated_data_path = None
    self.preaggregated_validation_data_path = None
    self.aggregated_tensors = None

    print('Max Dictionary Size = ' + str(max_dict_size))

    print('Column ID = ' + str(column_id))

    # Step 1: Check and load dict


    # Load from dict from cache if possible
    local_data_dir = dataset.get_local_data_dir()
    print('local_data_dir = ' + str(local_data_dir))
    if not os.path.exists(local_data_dir):
      os.makedirs(local_data_dir)
    local_dict_path_prefix = os.path.join(local_data_dir, 'dict_' + 
      type(self).__name__ + 
      '_dict' + str(max_dict_size))

    local_dict_vocab_path = local_dict_path_prefix + str(column_id) + '.vocab'
    local_dict_model_path = local_dict_path_prefix + str(column_id) + '.model'
    local_untokened_data_file = local_dict_path_prefix + str(column_id) + '.untoken'
    local_untokened_validation_data_file = local_dict_path_prefix + str(column_id) + '.valid.untoken'

    # We ensure that untokenized data file is available because we will use as inputs
    # to BERT example writer (For both training and validation dataset)
    if not os.path.exists(local_untokened_data_file) or not os.path.exists(local_dict_model_path):
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

      # Train sentence piece model (only on training data file)
      spm.SentencePieceTrainer.Train('--pad_id=0 --bos_id=2 --eos_id=3 --unk_id=1 --user_defined_symbols=<MASK> --input=' + 
        local_untokened_data_file + 
        ' --model_prefix=sp --vocab_size=' + str(max_dict_size) + ' --hard_vocab_limit=false')

      # Move sp.model / sp.vocab to the dict paths
      os.rename("sp.model", local_dict_model_path)
      os.rename("sp.vocab", local_dict_vocab_path)

      self.sentence_piece_processor.Load(local_dict_model_path)                  
    else:
      self.sentence_piece_processor.Load(local_dict_model_path)

    if not os.path.exists(local_untokened_validation_data_file):
      # Create untokened data file for validation dataset
      with open(local_untokened_validation_data_file, 'w', encoding='utf-8') as fout:
        print('Constructing untokened document')
        (_, _, x, y) = dataset.load_as_list()
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

    print('Dictionary size = ' +str(self.sentence_piece_processor.GetPieceSize()))

    # Step 2: Check and create data as 4 features set
    local_data_record_dir = os.path.join(local_data_dir, 'features_' + 
      type(self).__name__ + '_dict' + str(max_dict_size)) + str(column_id) + '_len' + str(config['max_seq_length'])
    self.preaggregated_data_path = local_data_record_dir
    if not os.path.exists(local_data_record_dir):
      print('[INFO] Start generating TFRecord file from untokenned data file at: ' + local_data_record_dir)
      example_writer = BERTSPMExampleWriter(
        job_id=0,
        spm_model=self.sentence_piece_processor,
        output_dir=local_data_record_dir,
        max_seq_length=config['max_seq_length'],
        num_jobs=1,
        blanks_separate_docs=True, # args.blanks_separate_docs,
        do_lower_case=True, # args.do_lower_case
        cls_id=2,
        sep_id=3,
        mask_id=4
      )      
      example_writer.write_examples(local_untokened_data_file)
      example_writer.finish()
      print('[INFO] Finished generating TFRecord (Training Dataset): ' + local_data_record_dir)

    local_validation_data_record_dir = os.path.join(local_data_dir, 'features_validation_' + 
      type(self).__name__ + '_dict' + str(max_dict_size)) + str(column_id) + '_len' + str(config['max_seq_length'])
    self.preaggregated_validation_data_path = local_validation_data_record_dir
    if not os.path.exists(local_validation_data_record_dir):
      print('[INFO] Start generating TFRecord file from untokenned data file at: ' + local_validation_data_record_dir)
      example_writer = BERTSPMExampleWriter(
        job_id=0,
        spm_model=self.sentence_piece_processor,
        output_dir=local_validation_data_record_dir,
        max_seq_length=config['max_seq_length'],
        num_jobs=1,
        blanks_separate_docs=True, # args.blanks_separate_docs,
        do_lower_case=True, # args.do_lower_case
        cls_id=2,
        sep_id=3,
        mask_id=4
      )      
      example_writer.write_examples(local_untokened_validation_data_file)
      example_writer.finish()
      print('[INFO] Finished generating TFRecord (Training Dataset): ' + local_validation_data_record_dir)

    # Step 3: Mask out some token and store as seperated label file

  def startid(self):  return 2
  def endid(self):    return 3
  def maskid(self):    return 4

  # Function used for encode batch of string data into batch of encoded integer
  def encode(self, token_list, max_length = 999):

    if 'max_seq_length' in self.config:
      max_length = self.config['max_seq_length']

    mask_last_token = False   
    if 'mask_last_token' in self.config:
      mask_last_token = self.config['mask_last_token']

    # This is to force placing special clf_id not exceed specific location (Such as len-1 in decoder only architecture because it trims the last token out)
    clf_id = None
    clf_pos_offset = None
    if 'clf_id' in self.config:
      clf_id = self.config['clf_id']
    if 'clf_pos_offset' in self.config:
      clf_pos_offset = self.config['clf_pos_offset']

    '''
    "input_ids": create_int_feature(input_ids),
    "input_mask": create_int_feature(input_mask),
    "segment_ids": create_int_feature(segment_ids)
    '''

    input_ids = np.zeros((len(token_list), max_length), dtype='int32')
    input_mask = np.zeros((len(token_list), max_length), dtype='int32')
    segment_ids = np.zeros((len(token_list), max_length), dtype='int32')

    input_ids[:,0] = self.startid() 
    for i, x in enumerate(token_list):
      x = x[:max_length - 1]
      x = self.trivial_token_separator.join(x).strip()
      encoded_x = self.sentence_piece_processor.EncodeAsIds(x)
      # sys.stdout.buffer.write(x.encode('utf8'))
      # Ensure that we are not 
      encoded_x = encoded_x[:max_length - 1]
      input_ids[i, 1:len(encoded_x) + 1] = encoded_x

      # If sentence is not end, then don't add end symbol at the end of encoded tokens
      # We have to mask out last token in some case (Language Model). Note that masked token can be endid() (predict end of sequence)
      if 1 + len(encoded_x) < max_length:
        if mask_last_token:
          input_ids[i, 1 + len(encoded_x)] = 0
          input_mask[i, 0:1 + len(encoded_x)] = 1
        else:
          input_ids[i, 1 + len(encoded_x)] = self.endid()
          input_mask[i, 0:1 + len(encoded_x)] = 1
      else:
        if mask_last_token:
          input_ids[i, len(encoded_x)] = 0      
          input_mask[i, 0:len(encoded_x)] = 1

      # If clf_pos_offset is specified, we trim data to the length and set clf_id at the position
      if clf_pos_offset is not None:
        clf_pos = min(1 + len(encoded_x), max_length - 1 + clf_pos_offset)
        input_ids[i, clf_pos] = clf_id
        input_ids[i, clf_pos + 1:] = 0

      # print('Encoded Ids = ' + str(input_ids[i,:]))

    X = [
      input_ids,
      input_mask,
      segment_ids,
    ]

    if self.config['is_input'] == True:
      return X
    else:
      return X[0] # We need only 'input_ids' for output side
  
  # Function used for decode batch of integers back to batch of string
  def decode(self, id_list):
    ret = []
    for i, x in enumerate(id_list):
      x = [int(n) for n in x]
      text = self.sentence_piece_processor.DecodeIds(x)
      ret.append(text)
    return ret

  # Function to return size of dictionary (key size)
  def num(self):
    return self.sentence_piece_processor.GetPieceSize()
  
  # Function to return list of objects to differentiate cached of input/output that model will use.
  # Basically it is configurations that effect encoded data.
  def get_data_effected_configs(self):
    mask_last_token = False   
    if 'mask_last_token' in self.config:
      mask_last_token = self.config['mask_last_token']

    clf_id = None
    clf_pos_offset = None
    if 'clf_id' in self.config:
      clf_id = self.config['clf_id']
    if 'clf_pos_offset' in self.config:
      clf_pos_offset = self.config['clf_pos_offset']
    clf_txt = ''
    if clf_pos_offset is not None:
      clf_txt = '_clf' + str(clf_id) + 'at' + str(clf_pos_offset)

    max_seq_length_txt = ''
    if 'max_seq_length' in self.config:
      max_seq_length_txt = '_len' + str(self.config['max_seq_length'])

    if mask_last_token:
      return  '_dict' + str(self.max_dict_size) + '_masklast' + clf_txt + max_seq_length_txt
    else:
      return '_dict' + str(self.max_dict_size) + '_' + clf_txt + max_seq_length_txt

  # This function returns dimention of data it consumes.
  # Ex: X = int[Count] => return 1
  # Ex: X = [int[Count], int[Count]] => return 2
  def get_data_dimension(self):
    if self.config["is_input"] == True:
      return 3 # [input_ids, input_mask, segment_ids]
    else:
      return 1 # Output also need only 'input_ids' tensors

  # Function indicates of the data transform has aggregated transformation applied on raw dataset or not.
  # Example is that BERT pretrained data transform will try to batch many lines of text from dataset.load_as_list()
  # into single data row to maximize length of tranformed dataset.
  # For such case, in model training, we should not use dataset.load_as_list() and call transform.encode one by one row
  # but instead we should load already transformed data. The flag is to indicate which loading approach to be used.
  # Note that encode/decode function should still be implemented because we will call it in online inference mode
  # or non-pretrained mode (ex, during finetuning)
  def is_data_preaggregated(self):
    if self.config['is_pretrain'] == True:
      return True
    else:
      return False

  # If data is pre-aggregated, this function is called to load pre-aggregated data instead of calling dataset.load_as_list().
  # Returns from this function should be (X, Y, X_valid, Y_valid) - or generator in future...
  def load_preaggregated_data(self):
    # Return objects of this function
    X = None
    Y = None
    X_valid = None
    Y_valid = None

    # Load pre-aggregated training dataset
    tfrecord_file_list = os.listdir(self.preaggregated_data_path)
    tfrecord_file_list = [os.path.join(self.preaggregated_data_path, k) for k in tfrecord_file_list]
    print('Pre-aggregated file list = ' + str(tfrecord_file_list))
    reader = tf.TFRecordReader()
    key, examples = reader.read(tf.train.string_input_producer(tfrecord_file_list, num_epochs=1)) # Only generate all data once

    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
    }    

    parsed_example = tf.parse_single_example(examples, name_to_features)
    parsed_example_values = list(parsed_example.values())

    # Reuse Keras Session
    sess = K.get_session()

    # Just read all data into array for now.
    # TODO: Implment generator to support very large dataset that is not fit into RAM
    all_data = []    
    sess.run(tf.initialize_local_variables())
    tf.train.start_queue_runners(sess=sess)
    try:
      while True:
        data = sess.run(parsed_example_values)
        for i in range(len(data)):
          if len(all_data) <= i:
            all_data.append([])
          all_data[i].append(data[i])
    except tf.errors.OutOfRangeError:
      pass
    all_data = [np.array(a) for a in all_data]
    X = all_data
    Y = all_data[0] # Y is only 'input_ids' tensor
    K.clear_session() # sess object is not valid anymore after this

    # Load pre-aggregated validation dataset
    tfrecord_file_list = os.listdir(self.preaggregated_validation_data_path)
    tfrecord_file_list = [os.path.join(self.preaggregated_validation_data_path, k) for k in tfrecord_file_list]
    print('Pre-aggregated file list = ' + str(tfrecord_file_list))
    reader = tf.TFRecordReader()
    key, examples = reader.read(tf.train.string_input_producer(tfrecord_file_list, num_epochs=1)) # Only generate all data once

    name_to_features = {
      "input_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([self.max_seq_length], tf.int64),
    }    

    parsed_example = tf.parse_single_example(examples, name_to_features)
    parsed_example_values = list(parsed_example.values())

    # Reuse Keras Session
    sess = K.get_session()

    # Just read all data into array for now.
    # TODO: Implment generator to support very large dataset that is not fit into RAM
    all_data = []
    sess.run(tf.initialize_local_variables())
    tf.train.start_queue_runners(sess=sess)
    try:
      while True:
        data = sess.run(parsed_example_values)
        for i in range(len(data)):
          if len(all_data) <= i:
            all_data.append([])
          all_data[i].append(data[i])
    except tf.errors.OutOfRangeError:
      pass
    all_data = [np.array(a) for a in all_data]
    X_valid = all_data
    Y_valid = all_data[0] # Y is only 'input_ids' tensor
    K.clear_session() # sess object is not valid anymore after this

    #print(len(X_valid))
    #print(len(Y_valid))

    return (X, Y, X_valid, Y_valid)

  # Function indicates if there is dynamic preprocessing needed to be applied on data or not.
  # Dynamic preprocessing is the logics those will be applied on data at starting of each epoch before feeding into to the model.
  # Example for such situation is "BERT" which we want to "mask" some tokens out, but we want it to be dynamically random in each eopch,
  # which mean for the same input string, we mask different tokens in each epoch of training.
  # This actually can be done once in data pre-aggregation step that create multiply dataset with different mask, 
  # or can be done here dynamically on-the-fly without need to multiple training data rows.
  def is_data_dynamically_aggregated(self):
    # We want to perform tokens random masking for input side only...
    if self.config["is_input"] == True:
      return True
    else:
      return False # Output also need only 'input_ids' tensors

  def scatter_update(self, sequence, updates, positions):
    """Scatter-update a sequence.

    Args:
      sequence: A [batch_size, seq_len] or [batch_size, seq_len, depth] tensor
      updates: A tensor of size batch_size*seq_len(*depth)
      positions: A [batch_size, n_positions] tensor

    Returns: A tuple of two tensors. First is a [batch_size, seq_len] or
      [batch_size, seq_len, depth] tensor of "sequence" with elements at
      "positions" replaced by the values at "updates." Updates to index 0 are
      ignored. If there are duplicated positions the update is only applied once.
      Second is a [batch_size, seq_len] mask tensor of which inputs were updated.
    """
    shape = self.get_shape_list(sequence, expected_rank=[2, 3])
    depth_dimension = (len(shape) == 3)
    if depth_dimension:
      B, L, D = shape
    else:
      B, L = shape
      D = 1
      sequence = tf.expand_dims(sequence, -1)
    N = self.get_shape_list(positions)[1]

    shift = tf.expand_dims(L * tf.range(B), -1)
    flat_positions = tf.reshape(positions + shift, [-1, 1])
    flat_updates = tf.reshape(updates, [-1, D])
    updates = tf.scatter_nd(flat_positions, flat_updates, [B * L, D])
    updates = tf.reshape(updates, [B, L, D])

    flat_updates_mask = tf.ones([B * N], tf.int32)
    updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask, [B * L])
    updates_mask = tf.reshape(updates_mask, [B, L])
    not_first_token = tf.concat([tf.zeros((B, 1), tf.int32),
                                tf.ones((B, L - 1), tf.int32)], -1)
    updates_mask *= not_first_token
    updates_mask_3d = tf.expand_dims(updates_mask, -1)

    # account for duplicate positions
    if sequence.dtype == tf.float32:
      updates_mask_3d = tf.cast(updates_mask_3d, tf.float32)
      updates /= tf.maximum(1.0, updates_mask_3d)
    else:
      assert sequence.dtype == tf.int32
      updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d))
    updates_mask = tf.minimum(updates_mask, 1)
    updates_mask_3d = tf.minimum(updates_mask_3d, 1)

    updated_sequence = (((1 - updates_mask_3d) * sequence) +
                        (updates_mask_3d * updates))
    if not depth_dimension:
      updated_sequence = tf.squeeze(updated_sequence, -1)

    return updated_sequence, updates_mask


  def _get_candidates_mask(self, all_inputs,
                          disallow_from_mask=None):
    """Returns a mask tensor of positions in the input that can be masked out."""
    input_ids, input_mask, segment_ids = all_inputs
    ignore_ids = [self.startid(), self.endid(), self.maskid()]
    candidates_mask = tf.ones_like(input_ids, tf.bool)
    for ignore_id in ignore_ids:
      candidates_mask &= tf.not_equal(input_ids, ignore_id)
    candidates_mask &= tf.cast(input_mask, tf.bool)
    if disallow_from_mask is not None:
      candidates_mask &= ~disallow_from_mask
    return candidates_mask

  def assert_rank(self, tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.

    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
      name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
      expected_rank_dict[expected_rank] = True
    else:
      for x in expected_rank:
        expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
      scope_name = tf.get_variable_scope().name
      raise ValueError(
          "For the tensor `%s` in scope `%s`, the actual rank "
          "`%d` (shape = %s) is not equal to the expected rank `%s`" %
          (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))

  def get_shape_list(self, tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.

    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if isinstance(tensor, np.ndarray) or isinstance(tensor, list):
      shape = np.array(tensor).shape
      if isinstance(expected_rank, six.integer_types):
        assert len(shape) == expected_rank
      elif expected_rank is not None:
        assert len(shape) in expected_rank
      return shape

    if name is None:
      name = tensor.name

    if expected_rank is not None:
      self.assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
      if dim is None:
        non_static_indexes.append(index)

    if not non_static_indexes:
      return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
      shape[index] = dyn_shape[index]
    return shape


  def mask(self, max_predictions_per_seq,
          all_inputs, mask_prob, proposal_distribution=1.0,
          disallow_from_mask=None, already_masked=None):
    """Implementation of dynamic masking. The optional arguments aren't needed for
    BERT/ELECTRA and are from early experiments in "strategically" masking out
    tokens instead of uniformly at random.

    Args:
      config: configure_pretraining.PretrainingConfig
      inputs: pretrain_data.Inputs containing input input_ids/input_mask
      mask_prob: percent of tokens to mask
      proposal_distribution: for non-uniform masking can be a [B, L] tensor
                            of scores for masking each position.
      disallow_from_mask: a boolean tensor of [B, L] of positions that should
                          not be masked out
      already_masked: a boolean tensor of [B, N] of already masked-out tokens
                      for multiple rounds of masking
    Returns: a pretrain_data.Inputs with masking added
    """
    input_ids, input_mask, segment_ids = all_inputs

    # Get the batch size, sequence length, and max masked-out tokens
    N = max_predictions_per_seq
    B, L = self.get_shape_list(input_ids)

    # Find indices where masking out a token is allowed
    candidates_mask = self._get_candidates_mask(all_inputs, disallow_from_mask)

    # Set the number of tokens to mask out per example
    num_tokens = tf.cast(tf.reduce_sum(input_mask, -1), tf.float32)
    num_to_predict = tf.maximum(1, tf.minimum(
        N, tf.cast(tf.round(num_tokens * mask_prob), tf.int32)))
    masked_lm_weights = tf.cast(tf.sequence_mask(num_to_predict, N), tf.float32)
    if already_masked is not None:
      masked_lm_weights *= (1 - already_masked)

    # Get a probability of masking each position in the sequence
    candidate_mask_float = tf.cast(candidates_mask, tf.float32)
    sample_prob = (proposal_distribution * candidate_mask_float)
    sample_prob /= tf.reduce_sum(sample_prob, axis=-1, keepdims=True)

    # Sample the positions to mask out
    sample_prob = tf.stop_gradient(sample_prob)
    sample_logits = tf.log(sample_prob)
    masked_lm_positions = tf.random.categorical(
        sample_logits, N, dtype=tf.int32)
    masked_lm_positions *= tf.cast(masked_lm_weights, tf.int32)

    # Get the ids of the masked-out tokens
    shift = tf.expand_dims(L * tf.range(B), -1)
    flat_positions = tf.reshape(masked_lm_positions + shift, [-1, 1])
    masked_lm_ids = tf.gather_nd(tf.reshape(input_ids, [-1]),
                                flat_positions)
    masked_lm_ids = tf.reshape(masked_lm_ids, [B, -1])
    masked_lm_ids *= tf.cast(masked_lm_weights, tf.int32)

    # Update the input ids
    replace_with_mask_positions = masked_lm_positions * tf.cast(
        tf.less(tf.random.uniform([B, N]), 0.85), tf.int32)
    inputs_ids, _ = self.scatter_update(
        input_ids, tf.fill([B, N], self.maskid() ),
        replace_with_mask_positions)

    return [tf.stop_gradient(inputs_ids),
          masked_lm_positions,
          masked_lm_ids,
          masked_lm_weights]

  # This function returns tensor operators in Keras layer form to perform dynamically aggregation on training data.
  # Note that this will be added to calculation graph for to perform the operations on each input before feeding to model.
  # (or append after model output in case of output transformation)
  # We cannot perform it outside calculation graph because it will be much more slower and will break Keras training loop.
  def get_dynamically_aggregation_layer(self, all_input_tensors):
    # We want to perform tokens random masking for input side only...
    if self.aggregated_tensors is not None:
      return self.aggregated_tensors

    if self.config["is_input"] == True:
      print(all_input_tensors)

      # If we are not in pretrained mode, just do not mask input.
      # Set masked_lm_positions, masked_lm_weights as None
      if self.config["is_pretrain"] == False:
        # Get the batch size, sequence length, and max masked-out tokens
        mask_prob = 0.15
        max_predictions_per_seq = int((mask_prob + 0.005) * self.max_seq_length)
        N = max_predictions_per_seq
        B, L = self.get_shape_list(all_input_tensors[0])
        null_masked_lm_ids = tf.zeros([B, N], dtype=tf.int32, name='null_masked_lm_ids')
        null_masked_lm_weights = tf.zeros([B, N], dtype=tf.float32, name='null_masked_lm_weights')
        self.aggregated_tensors = [*all_input_tensors, null_masked_lm_ids, null_masked_lm_weights]
        return self.aggregated_tensors

      def do_mask(all_inputs):
        input_ids, input_mask, segment_ids = all_inputs

        #input_ids = tf.Print(input_ids, ['input_ids', tf.shape(input_ids), input_ids], summarize=32)
        #input_mask = tf.Print(input_mask, ['input_mask', tf.shape(input_mask), input_mask], summarize=32)

        mask_prob = 0.15
        max_predictions_per_seq = int((mask_prob + 0.005) * self.max_seq_length)
        updated_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights = self.mask(max_predictions_per_seq, all_inputs, mask_prob)

        ''' For debugging purpose, assign fixed masked tokens
        updated_input_ids = tf.constant([[3, 6, 4 ,8, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=tf.int32)
        masked_lm_positions = tf.constant([[2, 4]], dtype=tf.int32)
        masked_lm_ids = tf.constant([[5, 9]], dtype=tf.int32)
        masked_lm_weights = tf.constant([[1.0, 1.0]], dtype=tf.float32)
        '''
        '''
        updated_input_ids = tf.Print(updated_input_ids, ['updated_input_ids', tf.shape(updated_input_ids), updated_input_ids], summarize=32)
        masked_lm_positions = tf.Print(masked_lm_positions, ['masked_lm_positions', tf.shape(masked_lm_positions), masked_lm_positions], summarize=32)
        masked_lm_ids = tf.Print(masked_lm_ids, ['masked_lm_ids', tf.shape(masked_lm_ids), masked_lm_ids], summarize=32)
        masked_lm_weights = tf.Print(masked_lm_weights, ['masked_lm_weights', tf.shape(masked_lm_weights), masked_lm_weights], summarize=32)
        '''

        return [updated_input_ids, input_mask, segment_ids, masked_lm_positions, masked_lm_weights]

      input_ids, input_mask, token_type_ids = all_input_tensors
      all_aggregated_tensors = Lambda(do_mask, name='bert_random_mask')([input_ids, input_mask, token_type_ids])
      self.aggregated_tensors = all_aggregated_tensors
      return all_aggregated_tensors
    else:
      # For output, we only need the 'input_ids' tensor
      self.aggregated_tensors = all_input_tensors
      return all_input_tensors

# Unit Test
print('-===================-')
print(__name__)
if __name__ == '__unittest__':
#if __name__ == '__main__' or __name__ == 'tensorflow.keras.initializers':
  print('=== UNIT TESTING ===')
  config = {
    "column_id": 0,
    "max_seq_length": 16,
    "is_input": True,
    "is_pretrain": True
  }
  from NLP_LIB.datasets.array_dataset_wrapper import ArrayDatasetWrapper
  dataset = ArrayDatasetWrapper({
    'values': [
      ['Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World'], # X
      ['Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World'], # Y
      ['Hella', 'Warld','aello', 'World','Hello', 'Uorld','Hello', 'WWrld','HellZ', 'World'], # X Valid
      ['Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World','Hello', 'World'], # Y Valid
    ]
  })

  # duplicate them

  transform = BERTSentencePiecePretrainWrapper(config, dataset)

  test_data = ['Hello', 'World']
  print('test_data = ' + str(test_data))

  encoded_data = transform.encode(test_data)
  print('encoded_data = ' + str(encoded_data))

  token_ids = encoded_data[0]
  print('token_ids = ' + str(token_ids))

  decoded_data = transform.decode(token_ids)
  print('decoded_data = ' + str(decoded_data))

  X, Y, X_valid, Y_valid = transform.load_preaggregated_data()

  X_ids = X[0]
  print('X_ids = ' + str(X_ids))
  decoded_X = transform.decode(X_ids)
  print('decoded_X = ' + str(decoded_X))

  X_valid_ids = X_valid[0]
  print('X_valid_ids = ' + str(X_valid_ids))
  decoded_X_valid = transform.decode(X_valid_ids)
  print('decoded_X_valid = ' + str(decoded_X_valid))

  print('Finished')

