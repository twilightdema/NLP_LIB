from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for BEST2010 Topic Dataset. It basically reuse GCS Dataset feature
class BEST2010LMDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'best2010lm'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/best2010/best2010.lm.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/best2010/best2010.lm.valid.txt'
    config['gcs_input_dict_path'] = 'https://storage.googleapis.com/atv_dataset/best2010/best2010_word.txt'
    config['gcs_output_dict_path'] = 'https://storage.googleapis.com/atv_dataset/best2010/best2010_word.txt'
    config['gcs_combined_dict_path'] = 'https://storage.googleapis.com/atv_dataset/best2010/best2010_word.txt'
    super(BEST2010LMDatasetWrapper, self).__init__(config)
