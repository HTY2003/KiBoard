import os
import numpy as np
import unidecode
import re
import tensorflow as tf
tf.enable_eager_execution()

from keras_preprocessing.text import Tokenizer
from nltk.corpus import gutenberg
from utils import Model, TrainingValues, clean_text

#---------- Recreating data used to train model ----------
text = " ".join(clean_text(gutenberg.raw(i)) for i in gutenberg.fileids()[:3])
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
encoded = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1
word2idx = tokenizer.word_index
idx2word = tokenizer.index_word

#---------- Restoring model from checkpoint ----------
model = Model(vocab_size, TrainingValues.EMBEDDING_DIM, TrainingValues.UNITS, TrainingValues.BATCH_SIZE)
optimizer = tf.train.AdamOptimizer()
checkpoint_dir = './training_checkpoints_1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

#---------- Obtaining predictions from model ----------
class NextWord:
    def predict(word, n=1):
        try:
            input_eval = [word2idx[word.lower()]]
            input_eval = tf.expand_dims(input_eval, 0)
            hidden = [tf.zeros((1, TrainingValues.UNITS))]
            predictions, hidden = model(input_eval, hidden)
            predicted_id = tf.nn.top_k(predictions[-1], 3)[1].numpy()
            result = [idx2word[i] for i in predicted_id]
            return result
        except KeyError:
            return []

print(NextWord.predict("great", 3))
