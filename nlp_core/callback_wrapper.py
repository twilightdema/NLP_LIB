class CallbackWrapper:

  # When initialize callback, we pass model object and configuration to constructor
  def __init__(self, config, execution_config, model, dataset, input_data_transform, output_data_transform):
    self.config = config
    self.model = model
    self.input_data_transform = input_data_transform
    self.output_data_transform = output_data_transform

  # This function should return keras Callback instance constructed from configuration of this object.
  def get_keras_callback(self):
    return None
