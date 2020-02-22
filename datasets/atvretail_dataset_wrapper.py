from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for ATVRetail Dataset. It basically reuse GCS Dataset feature
class ATVRetailDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'atvretail'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/atvretail/retail_translate_train.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/atvretail/retail_translate_test.txt'
    # config['gcs_output_dict_path'] = 'https://storage.googleapis.com/atv_dataset/atvretail/label_map.txt'
    super(ATVRetailDatasetWrapper, self).__init__(config)

  # Override function to Read data file.
  # For Multi-Label problem, Each class in Y (Label) is separated by ',' instead of space as in sequence-to-sequence.
  def read_data_file(self, filepath):
    # Read data file
    x = []
    y = []
    with open(filepath, 'r', encoding='utf8') as fin:
      for line in fin:
        line = line.strip()
        if len(line) > 0:
          tokens = line.split('\t')
          line_x = tokens[0].strip()
          words_x = line_x.split(' ')
          x.append(words_x)
          words_y = None
          if len(tokens) > 1:
            line_y = tokens[1].strip()
            words_y = line_y.split(',')
          else:
            # If no label
            words_y = []
          y.append(words_y)
    return (x, y)