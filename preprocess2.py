# pip3 install pandas

import os
import glob
import shutil
import tarfile
import pandas as pd
import urllib.request
import tensorflow as tf
import tensorflow.keras as keras
import tqdm


# Download dataset
DATASET_URL = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
DATASET_DIR = 'dataset'
DATASET_FILE_PATH = f'{DATASET_DIR}/aclImdb_v1.tar.gz'

if not os.path.isfile(DATASET_FILE_PATH):
    print(f'Downloading dataset into {DATASET_FILE_PATH} ...')
    with urllib.request.urlopen(DATASET_URL) as response, open(DATASET_FILE_PATH, 'wb') as out_file:
        shutil.copyfileobj(response, out_file)
else:
    print('Dataset already downloaded.')

# Untar the dataset archive
if not os.path.isdir(f'{DATASET_DIR}/aclImdb'):
    with tarfile.open(DATASET_FILE_PATH) as archive:
        print(f'Extracting "{DATASET_FILE_PATH}" to "{DATASET_DIR}" ...')
        archive.extractall(DATASET_DIR)
        print('Extraction finished.')
else:
    print('Dataset already extracted.')

# Load data from folders
TEST_FOLDER = f'{DATASET_DIR}/aclImdb/test'
TEST_POSITIVE_FOLDER = f'{TEST_FOLDER}/pos'
TEST_NEGATIVE_FOLDER = f'{TEST_FOLDER}/neg'

VOCAB_SIZE = 10_000
MAX_SENTENCE_LEN = 100

vocabulary = open(f'{DATASET_DIR}/aclImdb/imdb.vocab').read().split('\n')
tokenizer = tf.keras.preprocessing.text.Tokenizer(VOCAB_SIZE)
tokenizer.fit_on_texts(vocabulary)

# Dataset of positive reviews
ds_files_pos = glob.glob(f'{TEST_POSITIVE_FOLDER}/*.txt')
ds_texts_pos = map(lambda fn: open(fn).read().split(' ')[:MAX_SENTENCE_LEN], ds_files_pos)
ds_sequences_pos = tokenizer.texts_to_sequences(ds_texts_pos)

def generator():
    for el in ds_sequences_pos:
        yield (el, [1, 0])

ds = tf.data.Dataset.from_generator(generator, (tf.int64, tf.int64))

# Dataset of negative reviews
ds_files_neg = glob.glob(f'{TEST_NEGATIVE_FOLDER}/*.txt')
ds_texts_neg = map(lambda fn: open(fn).read().split(' ')[:MAX_SENTENCE_LEN], ds_files_neg)
ds_sequences_neg = tokenizer.texts_to_sequences(ds_texts_neg)

def gen_negative():
    for el in ds_sequences_neg:
        yield (el, [0, 1])

ds_neg = tf.data.Dataset.from_generator(gen_negative, (tf.int64, tf.int64))

def gen_all():
    g1 = gen_negative()
    g2 = generator()
    while True:
        val1 = next(g1, None)
        val2 = next(g2, None)

        if val1:
            yield val1

        if val2:
            yield val2

        if val2 == None and val1 == None:
            break

ds_whole = tf.data.Dataset.from_generator(gen_all, (tf.int64, tf.int64))

# Creating the whole dataset
# ds = ds.concatenate(ds_neg)
ds = ds_whole

ds = ds.shuffle(buffer_size=10_000)
ds = ds.apply(tf.data.experimental.bucket_by_sequence_length(
    lambda el, _: tf.size(el),
    [50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900],
    [32] * 15,
    padded_shapes=([None], [2])
))


# ds = ds.padded_batch(32, padded_shapes=([None], [2]), drop_remainder=True)

# for x,y in ds:
#     print(x, y)
#     input()

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+2, output_dim=64, mask_zero=True),
    tf.keras.layers.LSTM(128, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# model.compile(
#     optimizer=keras.optimizers.SGD(),
#     loss=keras.losses.BinaryCrossentropy()
# )

model.fit(ds, epochs=10)

print(model.predict(tokenizer.texts_to_sequences([["That", "movie", "sucked", "balls", "I", "could", "not", "enjoy", "a", "single", "second", "of", "it", "because", "of", "bad", "CGI"]])))

# optimizer = keras.optimizers.Adam(learning_rate=1e-5)
# loss_fn = keras.losses.CategoricalCrossentropy()

# for epoch in range(1):

#     ds = ds.shuffle(buffer_size=10_000)

#     for x, y in tqdm.tqdm(ds):
#         with tf.GradientTape() as tape:
#             logits = model(x)
#             loss_value = loss_fn(y, logits)

#         grads = tape.gradient(loss_value, model.trainable_weights)

#         optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # print('Training loss %s' % float(loss_value))

