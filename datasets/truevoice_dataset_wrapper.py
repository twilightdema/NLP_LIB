from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for Truevoice Dataset. It basically reuse GCS Dataset feature
class TruevoiceDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'truevoice'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/truevoice/truevoice.intent.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/truevoice/truevoice.intent.valid.txt'
    super(TruevoiceDatasetWrapper, self).__init__(config)
