# pip3 install pandas

import os
import shutil
import tarfile
import pandas as pd
import urllib.request
import tensorflow as tf

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

list_ds_pos = tf.data.Dataset.list_files(f'{TEST_POSITIVE_FOLDER}/*')
list_ds_neg = tf.data.Dataset.list_files(f'{TEST_NEGATIVE_FOLDER}/*')

vocabulary = open(f'{DATASET_DIR}/aclImdb/imdb.vocab').read().split('\n')
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(vocabulary)

def process_path(file_path):
    sequence = tokenizer.texts_to_sequences([tf.io.read_file(file_path)])
    label = 1
    return sequence, label
# process_path = lambda file_path: tf.io.read_file(file_path)

ds = list_ds_pos.map(process_path)

# Batch data
for batch in ds.batch(4).take(1):
  for seq, label in batch:
      print(seq, label)

# Batch padding
# vocabulary = open(f'{DATASET_DIR}/aclImdb/imdb.vocab').read().split('\n')
# tokenizer = tf.keras.preprocessing.text.Tokenizer()
# tokenizer.fit_on_texts(vocabulary)

# sequences = tokenizer.texts_to_sequences(t)
# padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')
# print(padded_sequences)


def download_data():
    ...


def create_dataset():
    ...

model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(input_dim=1000, output_dim=64))
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.fit(
    ds.make_one_shot_iterator(),
    epochs=1,
    verbose=1
)