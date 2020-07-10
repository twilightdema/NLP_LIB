from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for English Wikipedia Language Model Dataset. It basically reuse GCS Dataset feature
# The dataset was trimmed down to be in same magnitude as BEST2010 for comparison
class ENWIKISmallLMDatasetWrapper(GCSDatasetWrapper):
  
  def __init__(self, config):
    config['dataset_name'] = 'enwiki_small'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/enwiki/enwiki_small.lm.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/enwiki/enwiki_small.lm.valid.txt'
    super(ENWIKISmallLMDatasetWrapper, self).__init__(config)

  # Get token seperator (Ex. ' ' for English, '' for Thai - because no seperator)
  def get_trivial_token_separator(self):
    return ' '