# -*- encoding: utf-8 -*-
import os, codecs
import preprocessing


BASE_PATH = 'data'

AGENT_SESSION_PATH = os.path.join(BASE_PATH, 'agent_session.txt')
AGENT_NOPRE_PATH = os.path.join(BASE_PATH, 'agent_sequence.txt')
AGENT_PRE_PATH = os.path.join(BASE_PATH, 'agent_stemmed_sequence.txt')

VISITOR_SESSION_PATH = os.path.join(BASE_PATH, 'visitor_session.txt')
VISITOR_NOPRE_PATH = os.path.join(BASE_PATH, 'visitor_sequence.txt')
VISITOR_PRE_PATH = os.path.join(BASE_PATH, 'visitor_stemmed_sequence.txt')

LABEL_PATH = os.path.join(BASE_PATH, 'session_satisfaction.txt')


def load_data(preprocess, maxseq, encodeTime):
    '''

    :param preprocess: whether word sequences are stemmed
    :param maxseq: the number of maximum sequences which rnn takes as input
    :param encodeTime: whether
    :return:
    '''
    session_data = {}
    if preprocess:
        agent_path = AGENT_PRE_PATH
        visitor_path = VISITOR_PRE_PATH
    else:
        agent_path = AGENT_NOPRE_PATH
        visitor_path = VISITOR_PRE_PATH

    fvisitor = codecs.open(visitor_path, 'r', encoding='utf-8')
    fagent = codecs.open(agent_path, 'r', encoding='utf-8')

    visitor_lines = fvisitor.readlines()
    agent_lines = fagent.readlines()
    for ix, v_line in enumerate(visitor_lines):

        a_line = agent_lines[ix]
        v_data = v_line.strip().split()
        a_data = a_line.strip().split()
        v_session_id = v_data[0]
        a_session_id = a_data[0]

        if a_session_id != v_session_id:
            print('There are unknown errors on reading lines. line number:{}'.format(ix))
            exit()
        else:
            session_id = a_session_id

        v_target = v_data[1:]
        a_target = a_data[1:]
        words = []

        a_target_len = len(a_target)

        for wix, v_w in enumerate(v_target):
            if wix >= a_target_len:
                print('Unexpected error.')
                print(len(a_data))
                print(len(v_data))
                exit()
            a_w = a_target[wix]
            if a_w == v_w:
                # time indicator
                if encodeTime is True:
                    words.append(preprocessing.NULL_TOKEN)
                elif encodeTime is False:
                    pass

            elif a_w == preprocessing.NULL_TOKEN:
                words.append(v_w)
            elif v_w == preprocessing.NULL_TOKEN:
                words.append(a_w)

        session_data[session_id] = words[:maxseq]
    fvisitor.close()
    fagent.close()

    return session_data


def load_label():
    label_for_session = {}
    with open(LABEL_PATH, 'rt') as f:

        f.next()
        for line in f:

            data = line.strip().split('|')
            session_id = data[0]
            label = data[1].strip()
            label_for_session[session_id] = label

    return label_for_session
