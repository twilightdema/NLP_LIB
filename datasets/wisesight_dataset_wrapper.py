from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for Wisesight Dataset. It basically reuse GCS Dataset feature
class WisesightDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'wisesight'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/wisesight/converted_train.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/wisesight/converted_valid.txt'
    super(WisesightDatasetWrapper, self).__init__(config)
