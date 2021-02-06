import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from transformer_optimizer import GPTLearningRateSchedule, BERTLearningRateSchedule, create_bert_optimizer

gpt_learning_rate_schedule = GPTLearningRateSchedule(512)
bert_learning_rate_schedule = BERTLearningRateSchedule(               
               initial_learning_rate = 0.005,
               num_train_steps = 40000,
               warmup_steps = 10000,
               power=1
               )

plt.figure("GPT Learning Rate")
plt.plot(gpt_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()

plt.figure("BERT Learning Rate")
plt.plot(bert_learning_rate_schedule(tf.range(40000, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()

bert_optimizer = create_bert_optimizer(
                initial_learning_rate = 0.005,
                num_train_steps = 40000,
                warmup_steps = 10000)

print('Finished')
