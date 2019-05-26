import os
import time
import numpy as np
import unidecode
import re
import tensorflow as tf
tf.enable_eager_execution()

from keras_preprocessing.text import Tokenizer
from nltk.corpus import gutenberg
from utils import Model, TrainingValues, clean_text, loss_function

text = " ".join(clean_text(gutenberg.raw(i)) for i in gutenberg.fileids()[:3])
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
encoded = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

sequences = list()
for i in range(1, len(encoded)):
    sequence = encoded[i - 1:i + 1]
    sequences.append(sequence)

sequences = np.array(sequences)
X, Y = sequences[:, 0], sequences[:, 1]
X = np.expand_dims(X, 1)
Y = np.expand_dims(Y, 1)

dataset = tf.data.Dataset.from_tensor_slices((X, Y)).shuffle(TrainingValues.BUFFER_SIZE)
dataset = dataset.batch(TrainingValues.BATCH_SIZE, drop_remainder=True)

model = Model(vocab_size, TrainingValues.EMBEDDING_DIM, TrainingValues.UNITS, TrainingValues.BATCH_SIZE)
optimizer = tf.train.AdamOptimizer()
checkpoint_dir = './training_checkpoints_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

for epoch in range(TrainingValues.EPOCHS):
    start = time.time()
 
    hidden = model.reset_states()
 
    for (batch, (input, target)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            predictions, hidden = model(input, hidden)
 
            target = tf.reshape(target, (-1,))
            loss = loss_function(target, predictions)
 
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables))
 
            if batch % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, loss))
 
    if (epoch + 1) % 10 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

