import sys
sys.path.append('..')

import os
import glob
import shutil
import random
import tarfile
import argparse
import urllib.request
import tensorflow as tf
import numpy as np

# Process arguments
parser = argparse.ArgumentParser(description='This is a script which trains neural networks')
parser.add_argument('-bs', '--batch-size', dest='bs', type=int, help='Inputs batch size', required=True)
parser.add_argument('-vs', '--vocab-size', dest='vs', type=int, help='Vocabulary size', required=True)
parser.add_argument('-sl', '--sentence-length', dest='sl', type=int, help='Max length of sentences', required=True)
parser.add_argument('-pl', '--paragraph-length', dest='pl', type=int, help='Max length of paragraph in sentences', required=True)

parser.add_argument('-ld', '--log-dir', dest='ld', help='Directory to log data to in the logs directory', required=True)

parser.add_argument('-es', '--embedding-size', dest='es', type=int, help='Size of embeddings', required=False)
parser.add_argument('-hu', '--hidden-units', dest='hu', type=int, help='Size of hidden units', required=False)
parser.add_argument('-att', '--attention', dest='att', type=bool, help='Use attention', required=False)

parser.add_argument('-ep', '--epochs', dest='ep', type=int, help='Number of epochs', required=True)

args = parser.parse_args()

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

# Arguments
VOCAB_SIZE = args.vs
MAX_SENTENCE_LEN = args.sl  # Max words in a sentence
MAX_PAR_LEN = args.pl  # Max sentences in a review
BATCH_SIZE = args.bs

# Load data from folders
TEST_FOLDER = f'{DATASET_DIR}/aclImdb/test'
TEST_POSITIVE_FOLDER = f'{TEST_FOLDER}/pos'
TEST_NEGATIVE_FOLDER = f'{TEST_FOLDER}/neg'

TRAIN_FOLDER = f'{DATASET_DIR}/aclImdb/train'
TRAIN_POSITIVE_FOLDER = f'{TRAIN_FOLDER}/pos'
TRAIN_NEGATIVE_FOLDER = f'{TRAIN_FOLDER}/neg'
TRAIN_UNSUPERVISED_FOLDER = f'{TRAIN_FOLDER}/unsup'

def get_tokenizer(vocab_file, vocab_size, separator='\n'):
    vocab = open(vocab_file).read()
    tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size, split=separator, oov_token=1)  # last one is the oov, 0 is used for padding
    tokenizer.fit_on_texts([vocab])
    return tokenizer

tokenizer = get_tokenizer(f'{DATASET_DIR}/aclImdb/imdb.vocab', args.vs+1)

# Dataset
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
                         flat_labeled_files)
    return list(labeled_tokens)

# create_shifted_dataset_from_files([f'{TEST_POSITIVE_FOLDER}', f'{TEST_NEGATIVE_FOLDER}'])


def create_hierarchical_labeled_dataset_from_files(folders, label_map={'pos':[1, 0], 'neg': [0, 1]}, shuffle=True):
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
    labeled_texts = map(lambda example: [
        list(map(lambda x:
                 [x for x in
                     list(filter(None, x.translate({ord(c): None for c in '\'\"()?/\<>!@#$,;'})
                             .lower()
                             .split(' ')))[:MAX_SENTENCE_LEN]
                  if x
                  ],
                 open(example[0]).read().split('.'))),
        example[1]
    ], flat_labeled_files)

    # avg_sent_len = 0
    # counter = 0
    # for file in labeled_texts:
    #     counter += 1
    #     avg_sent_len += len(file[0])
    # print(avg_sent_len // counter)

    # max_sent_len = 0
    # counter = 0
    # for file in labeled_texts:
    #     counter += 1
    #     if len(file[0]) > max_sent_len:
    #         max_sent_len = len(file[0])
    # print(max_sent_len)

    # avg_sent_len = 0
    # counter = 0
    # for file in labeled_texts:
    #     for sentence in file[0]:
    #         counter += 1
    #         avg_sent_len += len(sentence)
    # print(avg_sent_len // counter)

    # max_sent_len = 0
    # for file in labeled_texts:
    #     for sentence in file[0]:
    #         if len(sentence) > max_sent_len:
    #             max_sent_len = len(sentence)
    # print(max_sent_len)

    # tokenize texts
    labeled_tokens = map(lambda example: [tf.keras.preprocessing.sequence.pad_sequences(
                                                tokenizer.texts_to_sequences(example[0]), MAX_SENTENCE_LEN, padding='post'),
                                            label_map[example[1]]], labeled_texts)

    return list(labeled_tokens), len(flat_labeled_files)


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

    return list(labeled_tokens), len(flat_labeled_files)

# create_hierarchical_labeled_dataset_from_files([f'{TEST_POSITIVE_FOLDER}',
# f'{TEST_NEGATIVE_FOLDER}, {TRAIN_POSITIVE_FOLDER}', f'{TRAIN_NEGATIVE_FOLDER}'])


cls_test_ds, num_test_samples = create_hierarchical_labeled_dataset_from_files([f'{TEST_POSITIVE_FOLDER}', f'{TEST_NEGATIVE_FOLDER}, {TRAIN_POSITIVE_FOLDER}', f'{TRAIN_NEGATIVE_FOLDER}'])
cls_train_ds, num_train_samples = create_hierarchical_labeled_dataset_from_files([f'{TRAIN_POSITIVE_FOLDER}', f'{TRAIN_NEGATIVE_FOLDER}'])


# generate examples in the form of ((bs, review)(bs, label))
def cls_test_gen():
    for el in cls_test_ds:
        yield (el[0][:MAX_PAR_LEN], el[1])


def cls_train_gen():
    for el in cls_train_ds:
        yield (el[0][:MAX_PAR_LEN], el[1])


ds_test = tf.data.Dataset.from_generator(lambda: cls_test_gen(),
                                        (tf.int64, tf.int64)).repeat()
ds_train = tf.data.Dataset.from_generator(lambda: cls_train_gen(),
                                        (tf.int64, tf.int64)).repeat()

ds_train = ds_train.padded_batch(
    BATCH_SIZE,
    padded_shapes=([MAX_PAR_LEN, None], [2]),
    drop_remainder=True)

ds_test = ds_test.padded_batch(
    BATCH_SIZE,
    padded_shapes=([MAX_PAR_LEN, None], [2]),
    drop_remainder=True)

import layers

emb = 32
lstm1 = 8
drop1 = 0.25
lstm2 = 8
drop2 = 0.25

def get_model():
    return tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+2, output_dim=emb, mask_zero=True),
            input_shape=(MAX_PAR_LEN, args.sl)),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.LSTM(lstm1, dropout=drop1, activation='sigmoid')),
        tf.keras.layers.LSTM(lstm2, dropout=drop2, activation='sigmoid'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])


model = get_model()

# Model Saving
checkpoint_path = "checkpoints/model-cp-{epoch:04d}-{val_accuracy:.2f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Load Latest
latest = tf.train.latest_checkpoint(checkpoint_dir)

if latest:
    print('Model loaded')
    model.load_weights(latest)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 monitor='val_accuracy',
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.summary()
model.fit(ds_train,
        epochs=args.ep,
        shuffle=True,
        validation_data=ds_test,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        validation_steps=num_test_samples // BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join("logs", f'emb{emb}-l1_{lstm1}-drop1_{drop1}-l2_{lstm2}-drop2_{drop2}-' + args.ld),
                histogram_freq=1,
                profile_batch=0),
            cp_callback
        ])
