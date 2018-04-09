# -*- encoding: utf-8 -*-
from utils import ModelType
from trainer import read_train_eval


if __name__ == '__main__':

    read_train_eval(testid="GRU_siggle", preprocess=True, maxseq=100, modelType=ModelType.GRU_RNN,
                    dropout=0.1, earlyStop=True, seedNum=20160430, batchSize=70, maxEpoch=100, topn=True)
