import numpy as np
import tensorflow as tf
from transformer import TransformerEncoder, TransformerBottleNeckEncoder, positional_encoding
from transformer_optimizer import create_bert_optimizer

# Hyper-parameters
D_MODELS = [16, 32, 64, 128]
N_HEADS = [16, 8, 4, 2]

BATCH_SIZE = 16
SEQ_LEN = 128
OUTPUT_CLASS = 2

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
    out = self.transformer(x, training, mask)
    out = out[:, 0, :] # Pooled output (Gather only position of CLS token)
    out = self.fnn(out)
    return out

# Try printing output size from example input
encoder = TransformerBottleNeckClassifier(D_MODELS, N_HEADS, 2, 128)
sample_output = encoder(
    tf.random.uniform((64, 43, 16)), False, None)
print(sample_output.shape)  # (batch_size, class_num)

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

# Create sample input, mask and label
test_input = tf.random.uniform((BATCH_SIZE, SEQ_LEN, D_MODELS[0]))
test_mask = np.ones((BATCH_SIZE, SEQ_LEN), dtype=float)
test_label = tf.random.uniform((BATCH_SIZE, OUTPUT_CLASS))

def train_step(input, mask, label):
    with tf.GradientTape() as tape:
        predictions = model(input, 
                        True, 
                        None)
        print('Predictions')
        print(predictions.shape)
        print('Label')
        print(label.shape)
        loss = loss_function(label, predictions)
        print('Loss')
        print(loss)


    gradients = tape.gradient(loss, model.trainable_variables)    
    print('Gradients')
    print(len(gradients))

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(accuracy_function(label, predictions))

# Try perform 1 step training
train_step(test_input, test_mask, test_label)

print('Finished.')
