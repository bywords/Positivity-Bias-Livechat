# -*- encoding: utf-8 -*-
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Embedding, Reshape
from keras.initializers import glorot_normal
from keras import regularizers
import keras.layers


class SIMPLE_RNN():

    def __init__(self, vocab_size, maxlen, dropout, embedding, embedding_dim):

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_matrix = embedding
        self.embedding_dim = embedding_dim

    def __call__(self):
        text = Input(shape=(self.maxlen,), dtype='int32', name='text')
        x = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size, input_length=self.maxlen,
                            weights=[self.embedding_matrix], trainable=False)(text)

        output = keras.layers.SimpleRNN(units=self.embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='sigmoid', name='main_output',
                            activity_regularizer=regularizers.l2(0.001))(output)
        model = Model(inputs=text,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


class GRU_RNN():

    def __init__(self, vocab_size, maxlen, dropout, embedding, embedding_dim):

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_matrix = embedding
        self.embedding_dim = embedding_dim

    def __call__(self):
        text = Input(shape=(self.maxlen,), dtype='int32', name='text')
        x = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size, input_length=self.maxlen,
                            weights=[self.embedding_matrix], trainable=False)(text)

        output = keras.layers.GRU(units=self.embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='sigmoid', name='main_output',
                            activity_regularizer=regularizers.l2(0.001))(output)
        model = Model(inputs=text,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model


class LSTM_RNN():

    def __init__(self, vocab_size, maxlen, dropout, embedding, embedding_dim):

        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.dropout = dropout
        self.embedding_matrix = embedding
        self.embedding_dim = embedding_dim

    def __call__(self):
        text = Input(shape=(self.maxlen,), dtype='int32', name='text')
        x = Embedding(output_dim=self.embedding_dim, input_dim=self.vocab_size, input_length=self.maxlen,
                            weights=[self.embedding_matrix], trainable=False)(text)

        output = keras.layers.LSTM(units=self.embedding_dim, dropout=self.dropout,
                                        recurrent_dropout=self.dropout)(x)
        # And finally we add the main logistic regression layer
        main_output = Dense(1, activation='sigmoid', name='main_output',
                            activity_regularizer=regularizers.l2(0.001))(output)
        model = Model(inputs=text,
                      outputs=main_output)
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        return model
