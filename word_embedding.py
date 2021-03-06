# -*- encoding: utf-8 -*-
import os,codecs
import gensim
import multiprocessing
import reader
from preprocessing import encode_time_bucket, NULL_TOKEN
from utils import EmbeddingType, DataType


BASE_PATH = 'embedding'
PRE_ALL_PATH = os.path.join(BASE_PATH,'pre_word2vec_all.model')
NOPRE_ALL_PATH = os.path.join(BASE_PATH,'nopre_word2vec_all.model')


class SequenceIterator:
    def __init__(self, sequences):
        self.sequences = sequences

    def __iter__(self):
        for sequence in self.sequences:
            yield sequence.split()


def train_embedding(preprocess, datatype):
    if preprocess:
        embedding_path = PRE_ALL_PATH
    else:
        embedding_path = NOPRE_ALL_PATH
    encode_time = True

    if datatype is not DataType.ALL:
        word_sequences = reader.load_data(preprocess, datatype, encode_time)
        sequences_for_training = []
        for idx, words in word_sequences.items():
            sentences = [s for s in ' '.join(words).replace(NULL_TOKEN, '').split('  ') if s != '']
            for s in sentences:
                sequences_for_training.append(s)
    else:
        sequences_for_training = []
        word_sequences = reader.load_data(preprocess, DataType.VISITOR, encode_time)
        for idx, words in word_sequences.items():
            sentences = [s for s in ' '.join(words).replace(NULL_TOKEN, '').split('  ') if s != '']
            for s in sentences:
                sequences_for_training.append(s)
        word_sequences = reader.load_data(preprocess, DataType.AGENT, encode_time)
        for idx, words in word_sequences.items():
            sentences = [s for s in ' '.join(words).replace(NULL_TOKEN, '').split('  ') if s != '']
            for s in sentences:
                sequences_for_training.append(s)

    print('data load completed. start training.')
    print(sequences_for_training[0:2])

    cores = multiprocessing.cpu_count()
    model = gensim.models.Word2Vec(sentences=SequenceIterator(sequences_for_training), size=50,
                                   sg=1, min_count=5, window=5, workers=cores)
    model.save(embedding_path)


def load_embedding(embeddingType):

    if embeddingType is EmbeddingType.NOPRE_ALL:
        model = gensim.models.Word2Vec.load(NOPRE_ALL_PATH).wv
    elif embeddingType is EmbeddingType.PRE_ALL:
        model = gensim.models.Word2Vec.load(PRE_ALL_PATH).wv
    else:
        print("Unexpected errors on loading embedding model - embeddingType:{}".format(embeddingType))
        exit()

    return model


if __name__ == '__main__':

    train_embedding(False, DataType.ALL)
    train_embedding(True, DataType.ALL)
