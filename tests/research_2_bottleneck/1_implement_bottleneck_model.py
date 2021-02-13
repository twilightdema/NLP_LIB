import numpy as np
import tensorflow as tf
import time
import random
from transformer import TransformerEncoder, TransformerBottleNeckEncoder, positional_encoding, create_padding_mask
from transformer_optimizer import create_bert_optimizer

# Hyper-parameters
D_MODELS = [16, 32, 64, 128]
N_HEADS = [16, 8, 4, 2]

BATCH_SIZE = 16
SEQ_LEN = 128
OUTPUT_CLASS = 2

EPOCHS = 20

# The modified version of Transformer layer that perform different head / d_model in rach layer.
# This is to immitate CNN structure that can extract abstract idea data from input.
class TransformerBottleNeck(tf.keras.layers.Layer):
  def __init__(self, d_model_list, num_heads_list, max_len, rate=0.1):
    super(TransformerBottleNeck, self).__init__()
    self.positional_encoding = positional_encoding(max_len, d_model_list[0])
    self.encoders = []
    for i in range(len(d_model_list) - 1):
        encoder = TransformerBottleNeckEncoder(d_model_list[i], d_model_list[i+1], num_heads_list[i], d_model_list[i+1]*2)
        self.encoders.append(encoder)

  def call(self, x, training, mask):
    seq_len = tf.shape(x)[1]
    x = x + self.positional_encoding[:, :seq_len, :]
    for encoder in self.encoders:
        x = encoder(x, training, mask)
    return x

# The Transformer with classification layer
class TransformerBottleNeckClassifier(tf.keras.Model):
  def __init__(self, d_model_list, num_heads_list, num_class_out, max_len, rate=0.1):
    super(TransformerBottleNeckClassifier, self).__init__()
    self.transformer = TransformerBottleNeck(d_model_list, num_heads_list, max_len)
    self.fnn = tf.keras.layers.Dense(num_class_out)

  def call(self, x, training, mask):
    mask = tf.cast(mask, tf.float32)
    out = self.transformer(x, training, mask)
    out = out[:, 0, :] # Pooled output (Gather only position of CLS token)
    out = self.fnn(out)
    return out

# Loss function for Multi-class Classifier
loss_object = tf.keras.losses.CategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)

def accuracy_function(real, pred):
    #print('real = ' + str(real))
    #print('pred = ' + str(pred))
    accuracies = tf.equal(tf.argmax(real, axis=1), tf.argmax(pred, axis=1))
    #print('accuracies = ' + str(accuracies))
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    #print('accuracies = ' + str(accuracies))
    #print('accuracies = ' + str(tf.reduce_mean(accuracies)))
    #exit(0)
    return tf.reduce_mean(accuracies)

def create_input_mask(input):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(input)
    return enc_padding_mask

# Create metric object to hold metrics history
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

# Create optimizer object
optimizer = create_bert_optimizer(
    0.005,
    20,
    5,
    power=1.0,
    weight_decay_rate=0.0,
    epsilon=1e-6,
    name=None
)

# Create the model
model = TransformerBottleNeckClassifier(D_MODELS, N_HEADS, OUTPUT_CLASS, SEQ_LEN)

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

# Simulate training data
test_input, test_label = simulate_training_data(SEQ_LEN, D_MODELS[0], 160)
print('Training Input: ' + str(test_input.shape))
print('Training Label: ' + str(test_label.shape))

# Create input mask (directly from embedded input space)
test_mask = np.zeros((160, 1, 1, SEQ_LEN), dtype=float)
print('Training Mask: ' + str(test_mask.shape))

# Create dataset from the numpy arrays
train_dataset = tf.data.Dataset.from_tensor_slices((test_input, test_mask, test_label))
train_batches = train_dataset.cache().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def train_step(input, mask, label):
    with tf.GradientTape() as tape:
        predictions = model(input, 
                        True, 
                        mask)
        loss = loss_function(label, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(accuracy_function(label, predictions))

# Perform training for EPOCHS
for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()

    for (batch, (x, mask, y)) in enumerate(train_batches):
        train_step(x, mask, y)

    if batch % 5 == 0:
        print(f'Epoch {epoch + 1} Batch {batch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    '''
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print (f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
    '''
    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

print('Finished.')
