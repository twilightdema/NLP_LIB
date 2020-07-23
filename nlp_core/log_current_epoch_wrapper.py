from NLP_LIB.nlp_core.callback_wrapper import CallbackWrapper
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import os, re

class LogCurrentEpochWrapper:

  def __init__(self, training_config, dir_suffix):

    class _K_LogCurrentEpochWrapper(Callback):
      def __init__(self):
        # Home of output directory (support multi-OS)
        output_dir = os.path.join(*re.split('/|\\\\', training_config['output_dir']))
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)

        # Checkpoint saving directory
        checkpoint_dir = os.path.join(output_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
          os.makedirs(checkpoint_dir)
        epoch_filepath = os.path.join(checkpoint_dir, 'last_epoch' + dir_suffix + '.txt')        
        self.save_path = epoch_filepath

      def get_current_epoch(self):
        current_epoch = 0
        if os.path.exists(self.save_path):
          with open(self.save_path, 'r', encoding='utf-8') as f:
            current_epoch = int(f.readline().strip())          
        return current_epoch

      def on_batch_begin(self, batch, logs = None):
        pass

      def on_epoch_begin(self, epoch, logs=None):
        with open(self.save_path, 'w', encoding='utf-8') as f:
          f.write(str(epoch))

      def on_epoch_end(self, epoch, logs=None):
        with open(self.save_path, 'w', encoding='utf-8') as f:
          f.write(str(epoch + 1))

    self.keras_callback = _K_LogCurrentEpochWrapper()

  def get_current_epoch(self):
    return self.keras_callback.get_current_epoch()

  # This function should return keras Callback instance constructed from configuration of this object.
  def get_keras_callback(self):
    return self.keras_callback
