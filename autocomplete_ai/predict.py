import tensorflow as tf
from data import vocab_to_int, int_to_vocab
from utils import clean_text
from train import build_graph
epochs = 100
batch_size = 128
num_layers = 3
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.95
keep_probability = 0.75

def text_to_ints(text):
    '''Prepare the text for the model'''
    text = clean_text(text)
    return [vocab_to_int[word] for word in text]

# Create your own sentence or use one from the dataset
text = "Spellin is difficult, whch is wyh you need to study everyday."
text = text_to_ints(text)

checkpoint = "./autocomplete2.ckpt"

model = build_graph(keep_probability, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction)

with tf.Session() as sess:
    # Load saved model
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    #Multiply by batch_size to match the model's input parameters
    answer_logits = sess.run(model.predictions, {model.inputs: [text]*batch_size,
                                                 model.inputs_length: [len(text)]*batch_size,
                                                 model.targets_length: [len(text)+1],
                                                 model.keep_prob: [1.0]})[0]


pad = vocab_to_int["<PAD>"]

print('\nText')
print('  Word Ids:    {}'.format([i for i in text]))
print('  Input Words: {}'.format("".join([int_to_vocab[i] for i in text])))

print('\nSummary')
print('  Word Ids:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format("".join([int_to_vocab[i] for i in answer_logits if i != pad])))
