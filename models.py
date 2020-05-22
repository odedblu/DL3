import numpy as np
from keras.layers import LSTM, Dense, concatenate, Input, Embedding, Dropout, BatchNormalization
from keras.models import Model
from keras.optimizers import Adagrad
from keras.losses import CategoricalCrossentropy
import data_preproccesing as dp


def build_model_1(ebmedding_w):
    lyrics_input = Input(shape=(1,), name='lyrics_input')
    midi_input = Input(shape=(297,), name='midi_input')
    lyrics_embedding_l = Embedding(ebmedding_w.shape[0], 300, weights=[ebmedding_w], input_length=1, trainable=False,
                                   name='lyrics_embedding')(lyrics_input)
    lstm_1 = LSTM(100)(lyrics_embedding_l)
    concat_l = concatenate([lstm_1,midi_input])
    dense_1 = Dense(2024)(concat_l)
    bn_l = BatchNormalization()(dense_1)
    dense_2 = Dense(1024)(bn_l)
    dropout_l = Dropout(0.1)(dense_2)
    output_l = Dense(ebmedding_w.shape[0],activation='softmax')(dropout_l)
    model = Model([lyrics_input,midi_input],output_l)
    model.compile(optimizer=Adagrad(),loss=CategoricalCrossentropy(),metrics=['acc'])
    return model
