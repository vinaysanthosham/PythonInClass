# # Word2Vec

# The code for this lecture is based off the great tutorial example from tensorflow!
# Walk through:
# https://www.tensorflow.org/tutorials/word2vec
# Raw Code: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

# # Step 0: Imports
import collections
import math
import os
import errno
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange
from collections import Counter
import tensorflow as tf

# # Step 1: The data.
data_dir = "data"
data_url = 'http://mattmahoney.net/dc/text8.zip'


def fetch_words_data(url=data_url, words_data=data_dir):
    # Make the Dir if it does not exist
    os.makedirs(words_data, exist_ok=True)

    # Path to zip file 
    zip_path = os.path.join(words_data, "words.zip")

    # If the zip file isn't there, download it from the data url
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)

    # Now that the zip file is there, get the data from it
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])

    # Return a list of all the words in the data source.
    return data.decode("ascii").split()


# Use Defaults (this make take awhile!!)
words = fetch_words_data()

# Total words
print("Total length of words is: ", len(words))

for w in words[9000:9040]:
    print(w, end=' ')


## Build Word Counts and Create Word Data and Vocab

def create_counts(vocab_size=50000):
    # Begin adding vocab counts with Counter
    vocab = [] + Counter(words).most_common(vocab_size)

    # Turn into a numpy array
    vocab = np.array([word for word, _ in vocab])

    dictionary = {word: code for code, word in enumerate(vocab)}
    data = np.array([dictionary.get(word, 0) for word in words])
    return data, vocab


vocab_size = 50000

# This may take awhile
data, vocabulary = create_counts(vocab_size=vocab_size)


# ## Function for Batches

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


data_index = 0
batch, labels = generate_batch(8, 2, 1)

# Size of the bath
batch_size = 128

# Dimension of embedding vector
embedding_size = 100


# How many words to consider left and right (the bigger, the longer the training)
skip_window = 1

# How many times to reuse an input to generate a label
num_skips = 2

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.

# Random set of words to evaluate similarity on.
valid_size = 8





# Only pick dev samples in the head of the distribution.
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Number of negative examples to sample.
num_sampled = 64

# Model Learning Rate
learning_rate = 0.2

# How many words in vocab
vocabulary_size = 50000

# ## TensorFlow Placeholders and Constants

tf.reset_default_graph()

# Input data.
train_inputs = tf.placeholder(tf.int32, shape=[None])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

# ### Variables

# Look up embeddings for inputs.
init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)

embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# ### NCE Loss

# Construct the variables for the NCE loss
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch.
# tf.nce_loss automatically draws a new sample of the negative labels each
# time we evaluate the loss.
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed,
                   num_sampled, vocabulary_size))

# ### Optimizer

# Construct the Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.2)
trainer = optimizer.minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keepdims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer.
init = tf.global_variables_initializer()

# # Session
# Usually needs to be quite large to get good results,
# training takes a long time!
num_steps = 5000

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("./graphs", sess.graph)
    average_loss = 0