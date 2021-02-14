import os
import shutil
import numpy as np
import tensorflow as tf
import time
import random
from transformer import create_padding_mask, TransformerBottleNeckClassifier, TransformerClassifier
from transformer_optimizer import create_bert_optimizer

# Experiment ID
EXPERIMENT_ID = '3'

# If we need to resume training from checkpoint
RESUME_FROM_CHECKPOINT = False

# Use bottle neck model or normal Transformer architecture
USE_BOTTLENECT_MODEL = True

# Hyper-parameters
D_MODELS = [80, 80, 80, 80]
N_HEADS = [16, 8, 4, 2]

BATCH_SIZE = 16
SEQ_LEN = 128
OUTPUT_CLASS = 2

EPOCHS = 30

PEAK_LEARNING_RATE = 0.001
WARMUP_PERCENTAGE = 0.3 

####################################################################
# FUNCTION FOR SETUP RANDOMSEED SO THAT EXPERIMENTS ARE REPRODUCIBLE
RANDOM_SEED = 7654
def setup_random_seed(seed_value):
  # Set `PYTHONHASHSEED` environment variable at a fixed value
  os.environ['PYTHONHASHSEED'] = str(seed_value)
  # Set `python` built-in pseudo-random generator at a fixed value
  random.seed(seed_value)
  # Set `numpy` pseudo-random generator at a fixed value
  np.random.seed(seed_value)
  # Set `tensorflow` pseudo-random generator at a fixed value
  tf.random.set_seed(random.randint(0, 65535))

setup_random_seed(RANDOM_SEED)

####################################################################
# DETECT GPU WITH LOWEST LOADED AND USE THE GPU FOR ENTIRE PROGRAM
NUM_GPUS = len(tf.config.experimental.list_physical_devices('GPU'))
USED_DEVICE = None
if NUM_GPUS == 0:
  USED_DEVICE = '/device:CPU:0'
else:
  import subprocess as sp
  _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
  COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
  memory_free_infos = _output_to_list(sp.check_output(COMMAND.split()))
  memory_free_info = memory_free_infos[1:]
  memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
  chosen_gpu = np.argmax(np.array(memory_free_values))
  print('[INFO] Use GPU: ' + str(chosen_gpu))
  USED_DEVICE = '/device:GPU:' + str(chosen_gpu)

with tf.device(USED_DEVICE):

    # Simulation function
    def calculate_y(x):
        sum = 0.0
        for i in range(len(x)):
            sum = sum + x[i][0]
        sum = sum / len(x)
        if sum > 0.5:
            return [0, 1]
        else:
            return [1, 0]

    def simulate_training_data(seq_len, d_model, num):
        xs = []
        ys = []
        for i in range(num):
            x = []
            for j in range(seq_len):
                xx = random.random()
                xx = [xx] * d_model
                x.append(xx)
            y = calculate_y(x)
            xs.append(x)
            ys.append(y)
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    d_input_embedding = None
    if USE_BOTTLENECT_MODEL:
        d_input_embedding = D_MODELS[0]
    else:
        d_input_embedding = max(D_MODELS)

    # Simulate training data
    train_input, train_label = simulate_training_data(SEQ_LEN, d_input_embedding, 160)
    print('Training Input: ' + str(train_input.shape))
    print('Training Label: ' + str(train_label.shape))
    # Create input mask (directly from embedded input space)
    train_mask = np.zeros((160, 1, 1, SEQ_LEN), dtype=float)
    print('Training Mask: ' + str(train_mask.shape))

    # Simulate validation data
    valid_input, valid_label = simulate_training_data(SEQ_LEN, d_input_embedding, 32)
    print('Validation Input: ' + str(valid_input.shape))
    print('Validation Label: ' + str(valid_label.shape))
    # Create input mask (directly from embedded input space)
    valid_mask = np.zeros((32, 1, 1, SEQ_LEN), dtype=float)
    print('Validation Mask: ' + str(valid_mask.shape))

    # Create dataset from the numpy arrays
    train_dataset = tf.data.Dataset.from_tensor_slices((train_input, train_mask, train_label))
    train_batches = train_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input, valid_mask, valid_label))
    valid_batches = valid_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Calculate training steps
    training_steps = int(train_input.shape[0] * EPOCHS // BATCH_SIZE)
    warmp_up_steps = int(training_steps * WARMUP_PERCENTAGE)
    print('[INFO] Training steps = ' + str(training_steps))
    print('[INFO] Warm Up steps = ' + str(warmp_up_steps))


    # Loss function for Multi-class Classifier
    loss_object = tf.keras.losses.CategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        loss_ = loss_object(real, pred)
        return tf.reduce_mean(loss_)

    def accuracy_function(real, pred):
        accuracies = tf.equal(tf.argmax(real, axis=1), tf.argmax(pred, axis=1))
        accuracies = tf.cast(accuracies, dtype=tf.float32)
        return tf.reduce_mean(accuracies)

    def create_input_mask(input):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(input)
        return enc_padding_mask

    def get_trainable_parameter_counts(model):
        total_parameters = 0
        for variable in model.trainable_variables:
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim
            total_parameters += variable_parameters
        return total_parameters

    # Create metric object to hold metrics history
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.Mean(name='valid_accuracy')

    # Create optimizer object
    optimizer, bert_lr = create_bert_optimizer(
        PEAK_LEARNING_RATE,
        training_steps,
        warmp_up_steps,
        power=1.0,
        weight_decay_rate=0.0,
        epsilon=1e-6,
        name=None
    )

    # Create the model
    model = None
    if USE_BOTTLENECT_MODEL:
        model = TransformerBottleNeckClassifier(D_MODELS, N_HEADS, OUTPUT_CLASS, SEQ_LEN)
    else:
        d_model = max(D_MODELS)
        n_heads = max(N_HEADS)
        model = TransformerClassifier(d_model, len(D_MODELS), n_heads, OUTPUT_CLASS, SEQ_LEN)

    checkpoint_path = os.path.join('checkpoints', EXPERIMENT_ID)
    ckpt = tf.train.Checkpoint(model=model,
                            optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)

    # If a checkpoint exists, restore the latest checkpoint.
    if RESUME_FROM_CHECKPOINT and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('[INFO] Init weights from checkpoint: ' + str(checkpoint_path))

    # Setup tensorboard log
    train_tfboard_log_dir = os.path.join('tfboards', EXPERIMENT_ID, 'train')
    valid_tfboard_log_dir = os.path.join('tfboards', EXPERIMENT_ID, 'valid')
    misc_tfboard_log_dir = os.path.join('tfboards', EXPERIMENT_ID, 'misc')
    if os.path.exists(train_tfboard_log_dir):
        shutil.rmtree(train_tfboard_log_dir)
    if os.path.exists(valid_tfboard_log_dir):
        shutil.rmtree(valid_tfboard_log_dir)
    if os.path.exists(misc_tfboard_log_dir):
        shutil.rmtree(misc_tfboard_log_dir)
    train_summary_writer = tf.summary.create_file_writer(train_tfboard_log_dir)
    valid_summary_writer = tf.summary.create_file_writer(valid_tfboard_log_dir)
    os.makedirs(misc_tfboard_log_dir)

    def train_step(input, mask, label):
        with tf.GradientTape() as tape:
            predictions, attention_weights = model(input, 
                            True, 
                            mask)
            loss = loss_function(label, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)    
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)
        train_accuracy(accuracy_function(label, predictions))

    def validation_step(input, mask, label):
        predictions, attention_weights = model(input, 
                        False, 
                        mask)
        loss = loss_function(label, predictions)
        valid_loss(loss)
        valid_accuracy(accuracy_function(label, predictions))

    # Perform training for EPOCHS
    total_parameters = None
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()
        for (batch, (x, mask, y)) in enumerate(train_batches):
            train_step(x, mask, y)
            if total_parameters is None:
                model.summary()
                total_parameters = get_trainable_parameter_counts(model)
                print('[INFO] Total Trainable Parameters = ' + str(total_parameters))
                log_path = os.path.join(misc_tfboard_log_dir, 'info.txt')
                with open(log_path, 'w', encoding='utf-8') as fout:
                    fout.write('total_parameters = ' + str(total_parameters))

        if batch % 5 == 0:
            print(f'Epoch {epoch + 1} Batch {batch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        ckpt_save_path = ckpt_manager.save()

        # Also log BERT learning rate
        learning_rate = bert_lr.lastest_lr.read_value()

        valid_loss.reset_states()
        valid_accuracy.reset_states()
        for x, mask, y in valid_batches:
            validation_step(x, mask, y)

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Acc {train_accuracy.result():.4f} VLoss {valid_loss.result():.4f} VAcc {valid_accuracy.result():.4f} LR {learning_rate:.4f} Elapse {time.time() - start:.2f}s')

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)
            tf.summary.scalar('learning_rate', learning_rate, step=epoch)
        with valid_summary_writer.as_default():
            tf.summary.scalar('loss', valid_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', valid_accuracy.result(), step=epoch)

    print('Finished.')
