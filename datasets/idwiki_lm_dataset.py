from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for Bahaza (Indo) Wikipedia Language Model Dataset. It basically reuse GCS Dataset feature
class IDWIKILMDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'idwiki'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/idwiki/idwiki.lm.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/idwiki/idwiki.lm.valid.txt'
    super(IDWIKILMDatasetWrapper, self).__init__(config)

  # Get token seperator (Ex. ' ' for English, '' for Thai - because no seperator)
  def get_trivial_token_separator(self):
    return ' '
    