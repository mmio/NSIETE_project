import sys
sys.path.append('..')

import os
import glob
import shutil
import random
import tarfile
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

# Arguments
VOCAB_SIZE = 1_000
MAX_SENTENCE_LEN = 30
BATCH_SIZE = 32

# Load data from folders
TEST_FOLDER = f'{DATASET_DIR}/aclImdb/test'
TEST_POSITIVE_FOLDER = f'{TEST_FOLDER}/pos'
TEST_NEGATIVE_FOLDER = f'{TEST_FOLDER}/neg'

TRAIN_FOLDER = f'{DATASET_DIR}/aclImdb/train'
TRAIN_POSITIVE_FOLDER = f'{TRAIN_FOLDER}/pos'
TRAIN_NEGATIVE_FOLDER = f'{TRAIN_FOLDER}/neg'
TRAIN_UNSUPERVISED_FOLDER = f'{TRAIN_FOLDER}/unsup'

def get_tokenizer(vocab_file, vocab_size, separator='\n'):
    vocab = open(vocab_file).read().split(separator)
    ## Is this right
    tokenizer = tf.keras.preprocessing.text.Tokenizer(vocab_size, oov_token=vocab_size)
    tokenizer.fit_on_texts(vocab)
    return tokenizer

tokenizer = get_tokenizer(f'{DATASET_DIR}/aclImdb/imdb.vocab', VOCAB_SIZE+1)

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

cls_test_ds, num_test_samples = create_labeled_dataset_from_files([f'{TEST_POSITIVE_FOLDER}', f'{TEST_NEGATIVE_FOLDER}, {TRAIN_POSITIVE_FOLDER}', f'{TRAIN_NEGATIVE_FOLDER}'])
cls_train_ds, num_train_samples = create_labeled_dataset_from_files([f'{TRAIN_POSITIVE_FOLDER}', f'{TRAIN_NEGATIVE_FOLDER}'])

def cls_test_gen():
    for el in cls_test_ds:
        yield (el[0], el[1])

def cls_train_gen():
    for el in cls_train_ds:
        yield (el[0], el[1])

ds_test = tf.data.Dataset.from_generator(lambda: cls_test_gen(),
                                        (tf.int64, tf.int64)).repeat()
ds_train = tf.data.Dataset.from_generator(lambda: cls_train_gen(),
                                        (tf.int64, tf.int64)).repeat()

ds_train = ds_train.padded_batch(
    BATCH_SIZE,
    padded_shapes=([None], [2]),
    drop_remainder=True)

ds_test = ds_test.padded_batch(
    BATCH_SIZE,
    padded_shapes=([None], [2]),
    drop_remainder=True)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=VOCAB_SIZE+2, output_dim=128, mask_zero=True),
    tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(48, activation='sigmoid')),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(ds_train,
        epochs=1,
        shuffle=True,
        validation_data=ds_test,
        steps_per_epoch=num_train_samples // BATCH_SIZE,
        validation_steps=num_test_samples // BATCH_SIZE,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join("logs", "baseline_small2"),
                histogram_freq=1,
                profile_batch=0)
        ])

# while True:
# 	print('Enter something:')
# 	inp = input()
# 	print(model.predict(tokenizer.texts_to_sequences([inp.split(' ')])))
