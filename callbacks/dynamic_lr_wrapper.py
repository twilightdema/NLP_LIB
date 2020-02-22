from NLP_LIB.nlp_core.callback_wrapper import CallbackWrapper
from keras.callbacks import Callback, K

class DynamicLearningRateWrapper(CallbackWrapper):

  def __init__(self, config, execution_config, model, dataset, input_data_transform, output_data_transform):
    super(DynamicLearningRateWrapper, self).__init__(config, execution_config, model, dataset, input_data_transform, output_data_transform)

    class _K_DynamicLearningRate(Callback):
      def __init__(self, d_model, warmup=4000, scale=1.0):
        self.basic = d_model**-0.5
        self.basic = self.basic * scale
        self.warm = warmup**-1.5

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

          (X, _, _, _) = model.load_encoded_data(dataset)
          training_sample_count = X.shape[0]
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
        lr = self.basic * min(self.step_num**-0.5, self.step_num*self.warm)
        # print('Setting Learning Rate to: ' + str(lr))
        K.set_value(self.model.optimizer.lr, lr)

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
