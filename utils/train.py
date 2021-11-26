#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import
from __future__ import print_function

import os
import time
import pickle
import random
import datetime
import numpy as np
import pandas as pd
import multiprocessing
import dask.dataframe as dd
from pymongo import MongoClient
import tensorflow.compat.v1 as tf
from sklearn.utils import class_weight
from uaitrain.arch.tensorflow import uflag
from tensorflow.keras.utils import Sequence
from tensorflow.compat.v1.keras import backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, preprocessing, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import warnings
warnings.filterwarnings('ignore')

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
session = tf.Session(config=config)
K.set_session(session)

flags = tf.app.flags

FLAGS = tf.app.flags.FLAGS

flags.DEFINE_string("input_train_file", None,
                    "train file as required, csv and xlsx format.")

flags.DEFINE_string("test_data_from", 'file', "test data from...")

flags.DEFINE_string("input_test_file", None, "test file, csv and xlsx format.")

flags.DEFINE_string("output_file", 'output_file.csv', "output file.")

flags.DEFINE_string("model_file", "shorten_urls.pb",
                    "model output file, pb fromat.")

flags.DEFINE_string("dump_file", "shorten_urls.pl",
                    "train data dump file, pl fromat.")

flags.DEFINE_integer("vocab_size", None,
                     "total of url and title vocab size as default.")

flags.DEFINE_integer("max_seq_length", None,
                     "toatal max url and title seq length as default.")

flags.DEFINE_integer("num_train_epochs", 100, "train steps, 100 as default.")

flags.DEFINE_integer("batch_size", 100, "train batch size, 32 as default.")

flags.DEFINE_bool("only_url", True,
                  "only uses url as input training, True as default.")

flags.DEFINE_bool("do_train", False, "train mode is on, False as default.")

flags.DEFINE_bool("do_eval", False, "validation mode is on, False as default.")

flags.DEFINE_bool("do_predict", False, "do prediction task, False as default.")


def loadData(source, file=None, uri=None, db=None, table=None, search_key=None):
    if source == 'file':
        try:
            data = pd.read_csv(file, encoding='utf-8')
        except:
            data = pd.read_csv(file, sep='\t', header=None, encoding='utf-8')
    elif source == 'mongo':
        client = MongoClient(uri)
        db = client[db]
        table = db[table]

        end_time = datetime.datetime.combine(
            datetime.datetime.now(), datetime.datetime.min.time()
        )
        start_time = end_time - datetime.timedelta(days=1)
        data = dd.from_pandas(pd.DataFrame(list(table.find({search_key: {
                              "$gte": start_time, "$lt": end_time}}))), npartitions=16 * multiprocessing.cpu_count())

    return data


def _get_seq_maxlen(texts, filters=None, lower=False, char_level=False):
    tokenizer = preprocessing.text.Tokenizer(
        filters=filters, lower=lower, char_level=char_level)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    sequence_maxlen = 0
    for sequence in sequences:
        if len(sequence) > sequence_maxlen:
            sequence_maxlen = len(sequence)

    return tokenizer, sequences, vocab_size, sequence_maxlen


def _get_url_info(data):
    urls = data['url'].str.split('://').str[1]
    url_tokenizer, url_sequences, url_vocab_size, url_seq_maxlen = _get_seq_maxlen(
        urls, char_level=True)

    return url_tokenizer, url_sequences, url_vocab_size, url_seq_maxlen


def _get_title_info(data):
    titles = data['title'].values
    with open('/data/origin/chinese_stop_words.txt', encoding='gbk') as f:
        chinese_stop_words = f.readlines()
    title_tokenizer, title_sequences, title_vocab_size, title_seq_maxlen = _get_seq_maxlen(
        [str(title) for title in titles], filters=chinese_stop_words[0])

    return title_tokenizer, title_sequences, title_vocab_size, title_seq_maxlen


def dataToMatrix(trainData, testData=None):
    assert(len(trainData) != 0)
    url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    if FLAGS.only_url:
        trainData = trainData[trainData['url'].notnull(
        ) & trainData['url'].str.match(url_pattern) & trainData['label'].notnull()]
        # trainData.drop_duplicates(inplace=True)
        if len(trainData) != 0:
            url_tokenizer, url_sequences, url_vocab_size, url_seq_maxlen = _get_url_info(
                trainData)
            with open(FLAGS.data_dir + FLAGS.dump_file, 'wb') as f:
                pickle.dump((url_tokenizer, url_seq_maxlen), f)
            with open(FLAGS.data_dir + FLAGS.dump_file, 'rb') as f:
                url_tokenizer, url_seq_maxlen = pickle.load(f)
            if testData is not None:
                testData = testData[testData['url'].notnull(
                ) & testData['url'].str.match(url_pattern)]
                urls = testData['url'].str.split('://').str[1]
                url_sequences = url_tokenizer.texts_to_sequences(urls)
        vocab_size = url_vocab_size
        max_seq_length = url_seq_maxlen
        texts = url_sequences
    else:
        trainData = trainData[trainData['url'].notnull(
        ) & trainData['title'].notnull() & trainData['url'].str.match(url_pattern) & trainData['label'].notnull()]
        trainData.drop_duplicates(inplace=True)
        if len(trainData) != 0:
            url_tokenizer, url_sequences, url_vocab_size, url_seq_maxlen = _get_url_info(
                trainData)
            title_tokenizer, title_sequences, title_vocab_size, title_seq_maxlen = _get_title_info(
                trainData)
            if testData is not None:
                testData = testData[testData['url'].notnull(
                ) & testData['title'].notnull() & testData['url'].str.match(url_pattern)]
                urls = testData['url'].str.split('://', expand=True, n=1)[1]
                titles = testData['title'].values
                url_sequences = url_tokenizer.texts_to_sequences(urls)
                title_sequences = title_tokenizer.texts_to_sequences(
                    [str(title) for title in titles])
        vocab_size = url_vocab_size + title_vocab_size
        max_seq_length = url_seq_maxlen + title_seq_maxlen
        url_title_sequences = []
        for url_sequence, title_sequence in zip(url_sequences, title_sequences):
            url_title_sequences.append(url_sequence + title_sequence)
        texts = url_title_sequences

    vocab_size = [FLAGS.vocab_size if FLAGS.vocab_size else vocab_size][0]
    max_seq_length = [
        FLAGS.max_seq_length if FLAGS.max_seq_length else max_seq_length][0]

    X = preprocessing.sequence.pad_sequences(
        texts, maxlen=max_seq_length, padding='post', truncating='post')

    if len(trainData) != 0 and testData is None:
        y = np.array(trainData['label'].values)
        return X, y, vocab_size, max_seq_length
    else:
        return X, testData


class Metrics(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(
            self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print('- val_precision: %.4f - val_recall %.4f - val_f1: %.4f' %
              (_val_precision, _val_recall, _val_f1))


def saveModel(X, y, vocab_size, max_seq_length):
    callbacks_list = [
        callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5),
        callbacks.ModelCheckpoint(
            filepath=FLAGS.output_dir + 'models/' + FLAGS.model_file,
            monitor='val_loss',
            save_best_only=True),
        callbacks.TensorBoard(
            log_dir=FLAGS.log_dir + 'logs/',
            histogram_freq=0,
            write_graph=True,
            write_images=True)]

    metrics = Metrics()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=random.randint(0, 100))

    train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    batch_size = FLAGS.batch_size
    train_data = train_data.batch(batch_size)
    val_data = val_data.batch(batch_size)

    # class_weights = class_weight.compute_class_weight('balanced',
    #                                                   np.unique(y),
    #                                                   y)

    strategy = tf.distribute.MirroredStrategy() 

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_data = train_data.with_options(options)
    val_data = val_data.with_options(options) 

    with strategy.scope():
        # model = models.Sequential()
        # model.add(layers.Embedding(vocab_size + 1, 128,
        #                            input_length=max_seq_length, trainable=True))
        # model.add(layers.Conv1D(64, 3, activation='relu'))
        # model.add(layers.GlobalMaxPooling1D())
        # model.add(layers.Dense(1, activation='sigmoid'))

        entry_input = layers.Input(shape=(max_seq_length,), name='entry_input')
        x = layers.Embedding(output_dim=128, input_dim=vocab_size + 1, input_length=max_seq_length)(entry_input)

        x = layers.Conv1D(64, 3, activation='relu')(x)
        x = layers.MaxPooling1D(2)(x)

        x = layers.Conv1D(64, 1, activation='relu')(x)
        x = layers.Conv1D(64, 3, activation='relu')(x)
        maxpooling = layers.MaxPooling1D(2)(x)

        for i in range(3):
            conv1d_1_0 = layers.Conv1D(64, 1, activation='relu')(x)

            conv1d_1_1 = layers.Conv1D(64, 1, activation='relu')(x)
            conv1d_1_3_1 = layers.Conv1D(64, 3, activation='relu')(conv1d_1_1)

            conv1d_1_2 = layers.Conv1D(64, 1, activation='relu')(x)
            conv1d_1_3_2 = layers.Conv1D(64, 3, activation='relu')(conv1d_1_2)
            conv1d_1_3_3_2 = layers.Conv1D(64, 3, activation='relu')(conv1d_1_3_2)

            x = layers.concatenate([maxpooling, conv1d_1_0, conv1d_1_3_1, conv1d_1_3_3_2], axis=1)

        x = layers.GlobalMaxPooling1D()(x)

        output = layers.Dense(1, activation='sigmoid', name='output')(x)

        model = models.Model(inputs=entry_input, outputs=output)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    if FLAGS.do_train and FLAGS.do_eval:
        model.fit(train_data, epochs=FLAGS.num_train_epochs, batch_size=FLAGS.batch_size,
                           validation_data=val_data, callbacks=callbacks_list, verbose=1)
    else:
        model.fit(train_data, epochs=FLAGS.num_train_epochs, batch_size=FLAGS.batch_size,
                           verbose=1)

    y_pred_proba = model.predict(X_val)
    y_pred = y_pred_proba.round()

    with open(FLAGS.output_dir + 'output/' + 'output_binary_val.txt', 'w') as f:
        f.write('val_acc: ' + str(accuracy_score(y_val, y_pred)) + '\n' +
                'val_precision: ' + str(precision_score(y_val, y_pred)) + '\n' +
                'val_recall: ' + str(recall_score(y_val, y_pred)) + '\n' +
                'val_f1: ' + str(f1_score(y_val, y_pred)) + '\n' +
                'val_auc: ' + str(roc_auc_score(y_val, y_pred_proba)))


def loadModel2Pred(X, testData):
    model_file = FLAGS.output_dir + 'models/' + FLAGS.model_file
    if model_file:
        model = models.load_model(model_file)
        y_pred_proba = model.predict(X, verbose=1)
        y_pred = y_pred_proba.round()

        if testData.columns.str.contains('label').any():
            y_true = testData['label']

            if len(y_true.unique()) != 1:
                with open(FLAGS.output_dir + 'output/' + 'output_binary_test.txt', 'w') as f:
                    f.write('Test ACC: ' + str(accuracy_score(y_true, y_pred)) + '\n' +
                            'Test Precision: ' + str(precision_score(y_true, y_pred)) + '\n' +
                            'Test Recall: ' + str(recall_score(y_true, y_pred)) + '\n' +
                            'Test F1: ' + str(f1_score(y_true, y_pred)) + '\n' +
                            'Test AUC: ' + str(roc_auc_score(y_true, y_pred_proba)))
            else:
                with open(FLAGS.output_dir + 'output/' + 'output_binary_test.txt', 'w') as f:
                    f.write('Test ACC: ' + str(accuracy_score(y_true, y_pred)) + '\n' +
                            'Test Precision: ' + str(precision_score(y_true, y_pred)) + '\n' +
                            'Test Recall: ' + str(recall_score(y_true, y_pred)) + '\n' +
                            'Test F1: ' + str(f1_score(y_true, y_pred)))

        testData['pred'] = y_pred
        testData['score'] = [(1 - y)[0] if y <= 0.5 else y[0]
                             for y in y_pred_proba]
        testData['pred'] = testData['pred'].astype('int').astype('str')
        testData['pred'] = testData['pred'].str.replace('1', '黑').str.replace('0', '白')
        testData.to_csv(FLAGS.output_dir + 'output/' + FLAGS.output_file, index=False)


def predict_One(url):
    model = models.load_model(FLAGS.output_dir + 'models/' + FLAGS.model_file)
    with open(FLAGS.data_dir + FLAGS.dump_file, 'rb') as f:
        url_tokenizer, url_seq_maxlen = pickle.load(f)
    texts = url_tokenizer.texts_to_sequences(url.split('://')[1])
    text = np.array([text[0] for text in texts]).reshape(1, -1)
    x = preprocessing.sequence.pad_sequences(
        text, maxlen=url_seq_maxlen, padding='post', truncating='post')

    y_pred_proba = model.predict(x)
    y_pred = y_pred_proba.round()

    label = y_pred[0][0]
    proba = y_pred_proba[0][0]
    score = [1 - proba if proba <= 0.5 else proba][0]

    return label, score


class SamplesSequence(Sequence):
    def __init__(self, x_set, batch_size):
        self.x = x_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array(batch_x)


def predict_on_batch(X, testData, epochs):
    model_file = FLAGS.output_dir + 'models/' + FLAGS.model_file
    if model_file:
        model = models.load_model(model_file)
        y_pred_proba = model.predict_generator(
            SamplesSequence(X, int(len(X) / epochs)), verbose=1)
        y_pred = y_pred_proba.round()

        if testData.columns.str.contains('label').any():
            y_true = testData['label']

            if len(y_true.unique()) != 1:
                with open(FLAGS.output_dir + 'output/' + 'output_binary_test.txt', 'w') as f:
                    f.write('Test ACC: ' + str(accuracy_score(y_true, y_pred)) + '\n' +
                            'Test Precision: ' + str(precision_score(y_true, y_pred)) + '\n' +
                            'Test Recall: ' + str(recall_score(y_true, y_pred)) + '\n' +
                            'Test F1: ' + str(f1_score(y_true, y_pred)) + '\n' +
                            'Test AUC: ' + str(roc_auc_score(y_true, y_pred_proba)))
            else:
                with open(FLAGS.output_dir + 'output/' + 'output_binary_test.txt', 'w') as f:
                    f.write('Test ACC: ' + str(accuracy_score(y_true, y_pred)) + '\n' +
                            'Test Precision: ' + str(precision_score(y_true, y_pred)) + '\n' +
                            'Test Recall: ' + str(recall_score(y_true, y_pred)) + '\n' +
                            'Test F1: ' + str(f1_score(y_true, y_pred)))

        testData['pred'] = y_pred
        testData['score'] = [(1 - y)[0] if y <= 0.5 else y[0]
                             for y in y_pred_proba]
        testData['pred'] = testData['pred'].astype('int').astype('str')
        testData['pred'] = testData['pred'].str.replace('1', '黑').str.replace('0', '白')
        testData.to_csv(FLAGS.output_dir + 'output/' + FLAGS.output_file, index=False)


def main(self):
    tf.logging.set_verbosity(tf.logging.INFO)

    train_file = FLAGS.data_dir + FLAGS.input_train_file

    if train_file:
        trainData = loadData('file', train_file)

    if FLAGS.test_data_from == 'file':
        test_file = FLAGS.data_dir + FLAGS.input_test_file
        testData = loadData('file', test_file)
    elif FLAGS.test_data_from == 'mongo':
        testData = loadData('mongo', uri='mongodb://lion:pangu_lion@173.1.17.125:27017',
                            db='lion', table='case_scan', search_key='c_at')
    else:
        testData = None

    if len(trainData) != 0 and FLAGS.do_train:
        X, y, vocab_size, max_seq_length = dataToMatrix(trainData)
        saveModel(X, y, vocab_size, max_seq_length)
        if len(testData) != 0 and FLAGS.do_predict:
            X, testData = dataToMatrix(trainData, testData)
            loadModel2Pred(X, testData)
    elif len(trainData) != 0 and len(testData) != 0 and FLAGS.do_predict:
        if len(testData) <= 500000:
            X, testData = dataToMatrix(trainData, testData)
            if len(testData) <= 100000:
                loadModel2Pred(X, testData)
            else:
                predict_on_batch(X, testData, 100000)
        else:
            url_pattern = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
            model_file = FLAGS.output_dir + 'models/' + FLAGS.model_file
            if model_file:
                model = models.load_model(model_file)
            with open(FLAGS.data_dir + FLAGS.dump_file, 'rb') as f:
                url_tokenizer, url_seq_maxlen = pickle.load(f)
            testData = testData[testData['url'].notnull() & testData['url'].str.match(
                url_pattern)]
            testData['pred'] = 0
            testData['score'] = 0
            for i in range(0, len(testData), 500000):
                print('Stage ' + str(i // 500000 + 1) + ':')
                testData_s = testData[i:i+500000]
                urls = testData_s['url'].str.split('://').str[1]
                url_sequences = url_tokenizer.texts_to_sequences(urls)
                x = preprocessing.sequence.pad_sequences(
                    url_sequences, maxlen=url_seq_maxlen, padding='post', truncating='post')
                if len(x) > 100000:
                    y_pred_proba = model.predict_generator(
                        SamplesSequence(x, int(len(x) / 100000)), verbose=1)
                else:
                    y_pred_proba = model.predict(x, verbose=1)
                y_pred = y_pred_proba.round()
                testData.loc[i:i+499999, 'pred'] = y_pred
                testData.loc[i:i+499999, 'score'] = [(1 - y)[0] if y <= 0.5 else y[0]
                                                     for y in y_pred_proba]
            testData['pred'] = testData['pred'].astype('int').astype('str')
            testData['pred'] = testData['pred'].str.replace('1', '黑').str.replace('0', '白')
            testData.to_csv(FLAGS.output_dir  + 'output/' + FLAGS.output_file, index=False)

    else:
        with open(FLAGS.output_dir + 'logs/' + 'output_' + str(time.strftime('%Y-%m-%d', time.localtime(time.time()))) + '.txt', 'w') as f:
            f.write(str(-1))


if __name__ == "__main__":
    flags.mark_flag_as_required("input_train_file")
    tf.app.run()
