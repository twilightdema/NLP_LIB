from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for BEST2010 Topic Dataset. It basically reuse GCS Dataset feature
class THWIKILMDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'thwiki'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/thwiki/thwiki.lm.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/thwiki/thwiki.lm.valid.txt'
    config['gcs_input_dict_path'] = 'https://storage.googleapis.com/atv_dataset/thwiki/thwiki_word.txt'
    config['gcs_output_dict_path'] = 'https://storage.googleapis.com/atv_dataset/thwiki/thwiki_word.txt'
    config['gcs_combined_dict_path'] = 'https://storage.googleapis.com/atv_dataset/thwiki/thwiki_word.txt'
    super(THWIKILMDatasetWrapper, self).__init__(config)
