import re
import tensorflow as tf
tf.enable_eager_execution()

from enum import IntEnum

class TrainingValues(IntEnum):
    BATCH_SIZE = 128
    EPOCHS = 2
    BUFFER_SIZE = 1000000
    EMBEDDING_DIM = 100
    UNITS = 512

class Model(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units, batch_size):
        super(Model, self).__init__()
        self.units = units
        self.batch_size = batch_size
 
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
 
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_activation='sigmoid',
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)
 
    def call(self, inputs, hidden):
        inputs = self.embedding(inputs)
        output, states = self.gru(inputs, initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, states

def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'\x1a', ' ', text)
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    for weird in ('~', 'æ', 'è', 'é', 'î','<', '`'):
        text = re.sub(weird, '', text)
    text = re.sub(' +',' ', text)
    return text.lower()

def loss_function(labels, logits):
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

