from NLP_LIB.datasets.gcs_dataset_wrapper import GCSDatasetWrapper

# Wrapper class for BEST2010 Topic Dataset. It basically reuse GCS Dataset feature
class BEST2010TopicDatasetWrapper(GCSDatasetWrapper):
  def __init__(self, config):
    config['dataset_name'] = 'best2010topic'
    config['gcs_train_data_path'] = 'https://storage.googleapis.com/atv_dataset/best2010/best2010.topic.txt'
    config['gcs_validation_data_path'] = 'https://storage.googleapis.com/atv_dataset/best2010/best2010.topic.valid.txt'
    super(BEST2010TopicDatasetWrapper, self).__init__(config)
