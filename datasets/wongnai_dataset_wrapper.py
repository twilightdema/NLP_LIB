from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for Wongnai Dataset. It basically reuse GCS Dataset feature
class WongnaiDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'wongnai'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/wongnai/converted_train.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/wongnai/converted_valid.txt'
    super(WongnaiDatasetWrapper, self).__init__(config)
