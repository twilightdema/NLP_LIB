import numpy as np
import os, sys
from NLP_LIB.nlp_core.data_transform_wrapper import DataTransformWrapper
import sentencepiece as spm
import random

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
    vocab = self._tokenizer.vocab
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
               num_out_files=1000):
    self._blanks_separate_docs = blanks_separate_docs
    self._example_builder = BERTSPMExampleBuilder(spm_model, cls_id, sep_id, mask_id, max_seq_length)
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

class BERTSentencePiecePretrainWrapper(DataTransformWrapper):
  
  # When initialize DataTransformWrapper, we pass configuration and dataset object to constructor
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

    print('Max Dictionary Size = ' + str(max_dict_size))

    print('Column ID = ' + str(column_id))

    # Step 1: Check and load dict

    # Load from dict from cache if possible
    local_data_dir = dataset.get_local_data_dir()
    print('local_data_dir = ' + str(local_data_dir))
    local_dict_path_prefix = os.path.join(local_data_dir, 'dict_' + 
      type(self).__name__ + 
      '_dict' + str(max_dict_size))

    local_dict_vocab_path = local_dict_path_prefix + str(column_id) + '.vocab'
    local_dict_model_path = local_dict_path_prefix + str(column_id) + '.model'
    local_untokened_data_file = local_dict_path_prefix + str(column_id) + '.untoken'

    # We ensure that untokenized data file is available because we will use as inputs
    # to BERT example writer...
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
        
      # Train sentence piece model
      spm.SentencePieceTrainer.Train('--pad_id=0 --bos_id=2 --eos_id=3 --unk_id=1 --user_defined_symbols=<MASK> --input=' + 
        local_untokened_data_file + 
        ' --model_prefix=sp --vocab_size=' + str(max_dict_size) + ' --hard_vocab_limit=false')

      # Move sp.model / sp.vocab to the dict paths
      os.rename("sp.model", local_dict_model_path)
      os.rename("sp.vocab", local_dict_vocab_path)

      self.sentence_piece_processor.Load(local_dict_model_path)                  
    else:
      self.sentence_piece_processor.Load(local_dict_model_path)

    print('Dictionary size = ' +str(self.sentence_piece_processor.GetPieceSize()))

    # Step 2: Check and create data as 4 features set
    local_data_record_path = os.path.join(local_data_dir, 'features_' + 
      type(self).__name__ + '_dict' + str(max_dict_size)) + str(column_id) + '.record'
    if not os.path.exists(local_data_record_path):
      example_writer = BERTSPMExampleWriter(
        job_id=0,
        spm_model=self.sentence_piece_processor,
        output_dir=args.output_dir,
        max_seq_length=config['max_seq_length'],
        num_jobs=1,
        blanks_separate_docs=True, # args.blanks_separate_docs,
        do_lower_case=True, # args.do_lower_case
      )      


    # Step 3: Mask out some token and store as seperated label file

  def startid(self):  return 2
  def endid(self):    return 3
  def maskid(self):    return 4

  # Function used for encode batch of string data into batch of encoded integer
  def encode(self, token_list, max_length = 999):
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

    X = np.zeros((len(token_list), max_length), dtype='int32')
    X[:,0] = self.startid() 
    for i, x in enumerate(token_list):
      x = x[:max_length - 1]
      x = self.trivial_token_separator.join(x).strip()
      encoded_x = self.sentence_piece_processor.EncodeAsIds(x)
      # sys.stdout.buffer.write(x.encode('utf8'))
      # Ensure that we are not 
      encoded_x = encoded_x[:max_length - 1]
      X[i, 1:len(encoded_x) + 1] = encoded_x

      # If sentence is not end, then don't add end symbol at the end of encoded tokens
      # We have to mask out last token in some case (Language Model). Note that masked token can be endid() (predict end of sequence)
      if 1 + len(encoded_x) < max_length:
        if mask_last_token:
          X[i, 1 + len(encoded_x)] = 0
        else:
          X[i,1 + len(encoded_x)] = self.endid()
      else:
        if mask_last_token:
          X[i, len(encoded_x)] = 0      

      # If clf_pos_offset is specified, we trim data to the length and set clf_id at the position
      if clf_pos_offset is not None:
        clf_pos = min(1 + len(encoded_x), max_length - 1 + clf_pos_offset)
        X[i, clf_pos] = clf_id
        X[i, clf_pos + 1:] = 0

      # print('Encoded Ids = ' + str(X[i,:]))

    return X
  
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

    if mask_last_token:
      return  '_dict' + str(self.max_dict_size) + '_masklast' + clf_txt
    else:
      return '_dict' + str(self.max_dict_size) + '_' + clf_txt
