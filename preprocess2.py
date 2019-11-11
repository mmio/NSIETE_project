# pip3 install pandas

import os
import glob
import shutil
import random
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

TRAIN_FOLDER = f'{DATASET_DIR}/aclImdb/train'
TRAIN_POSITIVE_FOLDER = f'{TRAIN_FOLDER}/pos'
TRAIN_NEGATIVE_FOLDER = f'{TRAIN_FOLDER}/neg'
TRAIN_UNSUPERVISED_FOLDER = f'{TRAIN_FOLDER}/unsup'

VOCAB_SIZE = 10_000
MAX_SENTENCE_LEN = 100

# Create tokenizer
def get_tokenizer(vocab_file, vocab_size, separator='\n'):
    # FIXME: filter out duplicates, don't use set -> nondeterministic sorting, this example is OK imdb.vocab has unique values
    vocab = open(vocab_file).read().split(separator) 
    tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size, oov_token=0)
    tokenizer.fit_on_texts(vocab)
    return tokenizer
tokenizer = get_tokenizer(f'{DATASET_DIR}/aclImdb/imdb.vocab', VOCAB_SIZE)

# Dataset of positive reviews
def create_shifted_dataset_from_files(folders, shuffle=True):
    files = map(lambda folder: glob.glob(f'{folder}/*'), folders)

    labeled_files = map(lambda files_per_folder:
                        map(lambda file_path:
                            [open(file_path).read().split(' ')[:-1], open(file_path).read().split(' ')[1:]]
                        , files_per_folder)
                    , files)

    flat_labeled_files = []
    for lf in labeled_files:
        for fl in lf:
            flat_labeled_files.append(fl)

    if shuffle:
        random.shuffle(flat_labeled_files)

    labeled_tokens = map(lambda example: [*tokenizer.texts_to_sequences([example[0]]),
                                          *tokenizer.texts_to_sequences([example[1]])],
                         flat_labeled_files);

# create_shifted_dataset_from_files([f'{TEST_POSITIVE_FOLDER}', f'{TEST_NEGATIVE_FOLDER}'])

def create_labeled_dataset_from_files(folders, label_map={'pos':[1, 0], 'neg': [0, 1]}, shuffle=True):
    files = map(lambda folder: [glob.glob(f'{folder}/*'), f'{folder}'], folders)

    # Assign label to every files based on folder they are in
    labeled_files = map(lambda files_with_label:
                        map(lambda file_path:
                            [file_path, files_with_label[1].split('/')[-1]] # Take only the last folde from the folder path
                        , files_with_label[0])
                    , files)

    # flatten list
    flat_labeled_files = []
    for lf in labeled_files:
        for fl in lf:
            flat_labeled_files.append(fl)

    if shuffle:
        random.shuffle(flat_labeled_files)

    # read file contents
    labeled_texts = map(lambda example: [open(example[0]).read().split(' ')[:MAX_SENTENCE_LEN], example[1]], flat_labeled_files)

    # tokenize texts
    labeled_tokens = map(lambda example: [*tokenizer.texts_to_sequences([example[0]]),
                                          label_map[example[1]]], labeled_texts)
    return labeled_tokens

cls_test_ds = create_labeled_dataset_from_files([f'{TEST_POSITIVE_FOLDER}', f'{TEST_NEGATIVE_FOLDER}'])
cls_train_ds = create_labeled_dataset_from_files([f'{TRAIN_POSITIVE_FOLDER}', f'{TRAIN_NEGATIVE_FOLDER}'])

def cls_test_gen():
    for el in cls_test_ds:
        yield (el[0], el[1])

def cls_train_gen():
    for el in cls_train_ds:
        yield (el[0], el[1])

ds_test = tf.data.Dataset.from_generator(cls_test_gen, (tf.int64, tf.int64))
ds_train = tf.data.Dataset.from_generator(cls_train_gen, (tf.int64, tf.int64))

##########################################################################################################################

ds_files_pos = glob.glob(f'{TRAIN_POSITIVE_FOLDER}/*.txt')
ds_texts_pos = map(lambda fn: open(fn).read().split(' ')[:MAX_SENTENCE_LEN], ds_files_pos)
ds_sequences_pos = tokenizer.texts_to_sequences(ds_texts_pos)

def generator():
    for el in ds_sequences_pos:
        yield (el, [1, 0])

ds = tf.data.Dataset.from_generator(generator, (tf.int64, tf.int64))

# Dataset of negative reviews
ds_files_neg = glob.glob(f'{TRAIN_NEGATIVE_FOLDER}/*.txt')
ds_texts_neg = map(lambda fn: open(fn).read().split(' ')[:MAX_SENTENCE_LEN], ds_files_neg)
ds_sequences_neg = tokenizer.texts_to_sequences(ds_texts_neg)

def gen_negative():
    for el in ds_sequences_neg:
        yield (el, [0, 1])

ds_neg = tf.data.Dataset.from_generator(gen_negative, (tf.int64, tf.int64))

# Creating the whole dataset
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
##########################################################################################################################

# ds = ds_whole
ds = ds_test
ds = ds.shuffle(buffer_size=10_000)
# Bucketing, how the fuck do I sort padded batches ?
ds = ds.apply(tf.data.experimental.bucket_by_sequence_length(
    lambda el, _: tf.size(el),
    [50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900],
    [32] * 15,
    padded_shapes=([None], [2]),
    drop_remainder=True
))

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+2, output_dim=128, mask_zero=True),
    tf.keras.layers.LSTM(48, activation='sigmoid'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(ds, epochs=1)

while True:
	print('Enter something:')
	inp = input()
	print(model.predict(tokenizer.texts_to_sequences([inp.split(' ')])))

