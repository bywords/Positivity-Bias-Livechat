# -*- encoding: utf-8 -*-
from __future__ import print_function
import os
import writer

from sklearn.model_selection import StratifiedShuffleSplit
from keras.preprocessing import sequence
from keras.callbacks import CSVLogger, EarlyStopping

from reader import load_data, load_label
from preprocessing import filter_labeled_data, transform_sequence, transform_label,\
    transform_labeled_data_listform, random_oversampling, get_wordvectors_from_keyedvectors, \
    load_topn_words, transform_sequence_using_topn
from word_embedding import load_embedding
from samsung_rnn import *
from utils import ModelType, EmbeddingType


def read_train_eval(testid, preprocess, maxseq, modelType, encodeTime,
                    dropout, earlyStop, seedNum, batchSize, maxEpoch, topn):
    '''

    :param testid: identifier
    :param preprocess: whether sequences are stemmed or not
    :param maxseq: the maximum sequence length
    :param modelType: one of SIMPLE_RNN | LSTM_RNN | GRU_RNN
    :param encodeTime:
    :param dropout
    :param earlyStop: whether training stops when errors are saturated
    :param seedNum: random seed
    :param batchSize
    :param maxEpoch
    :param topn: how frequently used word tokens are considered
    :return:
    '''
    N = 1000
    TRAIN_INSTANCE_DIR = os.path.join('log', '{}_{}_{}_{}_{}_{}_{}_{}_{}'
                                      .format(testid, preprocess, maxseq, modelType,
                                              dropout, earlyStop, seedNum, batchSize, maxEpoch))

    if not os.path.isdir(TRAIN_INSTANCE_DIR):
        os.mkdir(TRAIN_INSTANCE_DIR)
    log_csvfile = os.path.join(TRAIN_INSTANCE_DIR, 'log.csv')
    result_file = os.path.join(TRAIN_INSTANCE_DIR, 'results.txt')

    print('Load data')
    session_data = load_data(preprocess=preprocess, maxseq=maxseq, encodeTime=encodeTime)
    label_data = load_label()
    topN_words = load_topn_words(session_data, N)

    sequences, labels = filter_labeled_data(session_data, label_data)

    print('Load embedding')
    if preprocess:
        w2v_model = load_embedding(embeddingType=EmbeddingType.PRE_ALL)
    else:
        w2v_model = load_embedding(embeddingType=EmbeddingType.NOPRE_ALL)

    print('Pre-processing sequences')
    print(' - Get word vectors')
    vocab_size, embedding_dim, word_indices, embedding_matrix = \
        get_wordvectors_from_keyedvectors(w2v_model, seed=seedNum)

    print(' - Transform sequences')
    if topn is False:
        transformed_seq = transform_sequence(sequences, word_indices=word_indices)
    else:
        transformed_seq = transform_sequence_using_topn(sequences, word_indices, w2v_model, topN_words)

    print(' - Transform labels')
    transformed_labels = transform_label(label_data)
    print(' - Transform seq data to list')
    X, y = transform_labeled_data_listform(transformed_seq, transformed_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seedNum)
    for train_index, test_index in sss.split(X, y):
        pass

    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]

    X_train, y_train = random_oversampling(X_train, y_train, seed=seedNum)
    X_test, y_test = random_oversampling(X_test, y_test, seed=seedNum)

    X_train = sequence.pad_sequences(X_train, maxlen=maxseq)
    X_test = sequence.pad_sequences(X_test, maxlen=maxseq)

    list_callbacks = [CSVLogger(log_csvfile, separator=',', append=False)]
    if earlyStop:
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
        list_callbacks.append(earlyStopping)

    if modelType is ModelType.GRU_RNN:
        model = GRU_RNN(vocab_size=vocab_size, maxlen=maxseq, dropout=dropout,
                           embedding=embedding_matrix, embedding_dim=embedding_dim)()
        model.fit({'text': X_train}, y_train,
                  validation_data=({'text':X_test}, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict({'text':X_test}, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.LSTM_RNN:
        model = LSTM_RNN(vocab_size=vocab_size, maxlen=maxseq, dropout=dropout,
                           embedding=embedding_matrix, embedding_dim=embedding_dim)()
        model.fit({'text': X_train}, y_train,
                  validation_data=({'text': X_test}, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict({'text': X_test}, batch_size=batchSize, verbose=1)
    elif modelType is ModelType.SIMPLE_RNN:
        model = SIMPLE_RNN(vocab_size=vocab_size, maxlen=maxseq, dropout=dropout,
                                 embedding=embedding_matrix, embedding_dim=embedding_dim)()
        model.fit({'text': X_train}, y_train,
                  validation_data=({'text': X_test}, y_test),
                  batch_size=batchSize, epochs=maxEpoch, verbose=1, callbacks=list_callbacks)
        y_pred = model.predict({'text': X_test}, batch_size=batchSize, verbose=1)
    else:
        print('This function should be set for XXX_single modeltype.')
        exit()

    print('Evaluation..')
    with open(result_file, 'wt') as f:
        writer.eval(y_pred, y_test, file=f)

