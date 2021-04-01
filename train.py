'''
###训练代码###
'''
import numpy as np
import pandas as pd
import os
import json
from keras.layers import *
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from bert4keras.backend import keras, K, search_layer
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score as fs
from sklearn.metrics import classification_report
import Levenshtein
import datetime

np.random.seed(1998)

def prejson(input_path):
    tmp = []
    for line in open(input_path, 'r'):
        tmp.append(json.loads(line))

    data = pd.DataFrame(tmp)
    return data

train_df = pd.DataFrame()
for i in['长','短']:
    for j in['短','长']:
        for p in ['A','B']:
                train = prejson('souhu/data/'+i+j+'匹配'+p+'类/train.txt')
                #test = prejson('souhu/data/'+i+j+'匹配'+p+'类/test_with_id.txt')
                dev = prejson('souhu/data/'+i+j+'匹配'+p+'类/valid.txt')
                train_df = pd.concat([train,dev,train_df], axis=0, ignore_index=True)


train_df['labelA'] = train_df['labelA'].fillna(0).astype(int)
train_df['labelB'] = train_df['labelB'].fillna(0).astype(int)

valid_df['labelA'] = valid_df['labelA'].fillna(0).astype(int)
valid_df['labelB'] = valid_df['labelB'].fillna(0).astype(int)

train_df['label'] = train_df['labelA'] + train_df['labelB']
valid_df['label'] = valid_df['labelA'] + valid_df['labelB']

train_df.drop(["labelA", "labelB"], axis=1, inplace=True)
valid_df.drop(["labelA", "labelB"], axis=1, inplace=True)


train_data = train_df[['source', 'target', 'label']].values
valid_data = valid_df[['source', 'target', 'label']].values


def build_model(mode='bert', filename='bert', LR=1e-5, DR=0.3):
    path = '/content/drive/MyDrive/pre_bert/External/'+filename+'/'
    config_path = path+'bert_config.json'
    checkpoint_path = path+'bert_model.ckpt'
    dict_path = path+'vocab.txt'

    global tokenizer
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    bert = build_bert_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        with_pool=True,
        model=mode,
        return_keras_model=False,
    )
    
    output = bert.model.output

    output = Dropout(rate=DR)(output)
    output = Dense(units=2,
                   activation='softmax',
                   kernel_initializer=bert.initializer)(output)

    model = Model(bert.model.input, output)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=Adam(LR),
        metrics=['accuracy'],
    )
    return model


class data_generator(object):
    def __init__(self, data, batch_size=32, random=True):
        self.data = data
        self.batch_size = batch_size
        self.random = random
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self, random=False):
        idxs = list(range(len(self.data)))
        if random:
            np.random.shuffle(idxs)
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for i in idxs:
            text1, text2, label = self.data[i]
            token_ids, segment_ids = tokenizer.encode(
                text1, text2, max_length=128)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or i == idxs[-1]:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

    def forfit(self):
        while True:
            for d in self.__iter__(self.random):
                yield d




def do_train(mode='bert', filename='roberta', lastfour=False, LR=1e-5, DR=0.2, ext=False, batch_size=16):

    skf = StratifiedKFold(5, shuffle=True, random_state=2020)
    nfold = 1

    data = np.concatenate([train_data, valid_data], axis=0)

    for train_index, valid_index in skf.split(data[:, :2], data[:, 2:].astype('int')):
        train = data[train_index, :]
        valid = data[valid_index, :]

        train_generator = data_generator(train, batch_size)
        valid_generator = data_generator(valid, batch_size)

        model = build_model(mode=mode, filename=filename,
                            lastfour=lastfour, LR=LR, DR=DR)


        early_stopping = EarlyStopping(
            monitor='val_loss', patience=1, verbose=1)

        checkpoint = ModelCheckpoint('/content/drive/MyDrive/study/main/model_data/' + filename + '_weights/' + str(nfold) + '.weights',
                                         monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1)

        model.fit_generator(train_generator.forfit(),
                            steps_per_epoch=len(train_generator),
                            epochs=5,
                            validation_data=valid_generator.forfit(),
                            validation_steps=len(valid_generator),
                            callbacks=[early_stopping, checkpoint],
                            verbose=2,
                            )

        del model
        K.clear_session()
        nfold += 1




model = build_model(mode='bert', filename='roberta', lastfour=False)
do_train(mode='bert', filename='roberta', lastfour=False, LR=1e-6, batch_size=8)
