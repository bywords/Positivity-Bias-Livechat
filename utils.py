# -*- encoding:utf-8 -*-

# Needs to specify threshold values for time gap features
AGENT_FIRST_THRESHOLD = 12
AGENT_SECOND_THRESHOLD = 49
VISITOR_FIRST_THRESHOLD = 13
VISITOR_SECOND_THRESHOLD = 51


class EmbeddingType:

    NOPRE_ALL = 2
    PRE_ALL = 5


class DataType:

    ALL = 1
    AGENT = 2
    VISITOR = 3


class ModelType:

    SIMPLE_RNN = 1
    LSTM_RNN = 2
    GRU_RNN = 3
