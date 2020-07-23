from NLP_LIB.nlp_core.callback_wrapper import CallbackWrapper
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

# This code implements BERT-like learning rate

class BERTLearningRateWrapper(CallbackWrapper):

  def __init__(self, config, execution_config, model, dataset, input_data_transform, output_data_transform):
    super(BERTLearningRateWrapper, self).__init__(config, execution_config, model, dataset, input_data_transform, output_data_transform)

    class _K_DynamicLearningRate(Callback):
      def __init__(self, d_model, warmup=4000, scale=1.0):
        self.basic = 1e-4
        self.basic = self.basic * scale
        self.warm = warmup

        # If will init step num from intial epoch of model
        # step_num = epoch x (training_data_count / batch_size)
        self.step_num = 0      
        self.lazy_init = False
        self.execution_config = execution_config

      def initialize_step_num(self):
        initial_epoch = self.execution_config['initial_epoch']
        print('DynamicLearningRateWrapper ====> INIT') 
        if initial_epoch > 0:
          print('DynamicLearningRateWrapper ====> INIT_FROM_EPOCH') 
          batch_size = 1
          if 'batch_size' in execution_config and execution_config['batch_size'] is not None:
            batch_size = execution_config['batch_size']

          training_sample_count = 32
          #try:
          (X, _, _, _) = model.load_encoded_data(dataset)
          training_sample_count = X.shape[0]
          #except:
          #  pass
          print('Training Sample Count = ' + str(training_sample_count))

          self.step_num = initial_epoch * (training_sample_count // batch_size + (training_sample_count % batch_size > 0))
          print('Init Step Num ' + str(self) + ' from epoch: ' + str(initial_epoch) 
            + ', batch_size: ' + str(batch_size) + ' => step_num: ' + str(self.step_num))

      def on_batch_begin(self, batch, logs = None):        
        # Lazy init step_num
        if self.lazy_init == False:
          self.initialize_step_num()
          self.lazy_init = True

        # print('DynamicLearningRateWrapper->on_batch_begin: ')
        self.step_num += 1
        if self.step_num < self.warm:
          lr = self.step_num * self.basic / self.warm
        else:
          lr = self.basic
        # print('Setting Learning Rate to: ' + str(lr))
        #try:
        K.set_value(self.model.optimizer.lr, lr)
        #except:
        #  pass
        self.effective_lr = lr

      def on_epoch_begin(self, epoch, logs = None):
        # print('DynamicLearningRateWrapper->on_epoch_begin: ' + str(epoch))
        pass
    
    scale = 1.0
    if 'scale' in config:
      scale = config['scale']      
    self.keras_callback = _K_DynamicLearningRate(config['d_model'], config['warmup'], scale)

  # This function should return keras Callback instance constructed from configuration of this object.
  def get_keras_callback(self):
    return self.keras_callback

# Unit Test
if __name__ == '__main__':
  import cv2
  import matplotlib.pyplot as plt
  config = {
    'd_model': 512,
    'warmup': 10000,
    'scale': 1.0,
  }
  exec_config = {
    'initial_epoch': 1,
    'batch_size': 32,
  }
  dlr = BERTLearningRateWrapper(config, exec_config, None, None, None, None)
  cb = dlr.keras_callback
  lrs = []
  eps = []
  for i in range(50000):
    cb.on_batch_begin(None)
    lr = cb.effective_lr
    print('lr[' + str(i) + '] = ' + str(lr))
    eps.append(i)
    lrs.append(lr)

  plt.plot(eps, lrs)
  plt.show()

  print('Finished.')
