import pretty_midi
import models as m
import pickle as pkl
import data_preproccesing as dp
import numpy as np
import pandas as pd
import os
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model


def train_model_1():
    """
    train the first model
    :return:
    """
    vocabulary_size = dp.get_vocabulary_size()
    x, one_hot_y, instance_to_song, song_indexes, word_indexer = dp.prepare_set(vocabulary_size)
    midi_data_dict = pkl.load(open('midi_vectors.pkl', 'rb'))  # dp.get_midi_vectors()
    x_lyrics_train, x_melody_train, x_lyrics_val, x_melody_val, y_train, y_val = dp.split_train_validation(x, one_hot_y,
                                                                                                           instance_to_song,
                                                                                                           midi_data_dict)
    embadding = pkl.load(open('emb_w.pkl', 'rb'))
    lyrics_model = m.build_model_1(embadding)
    rlop = ReduceLROnPlateau(min_delta=0.01)
    mcp = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
    history = lyrics_model.fit([np.array(x_lyrics_train), np.array(x_melody_train)], np.array(y_train), epochs=100,
                               batch_size=512,
                               validation_data=([np.array(x_lyrics_val), np.array(x_melody_val)], np.array(y_val)),
                               verbose=2, callbacks=[rlop, mcp])
    lyrics_model.save('model_1.h5')
    return history, word_indexer


def train_model_2():
    """
    train the second model
    :return:
    """
    vocabulary_size = dp.get_vocabulary_size()
    x, one_hot_y, instance_to_song, song_indexes, word_indexer = dp.prepare_set(vocabulary_size)
    midi_data_dict = pkl.load(open('encoded_midi_vectors', 'rb'))  # dp.get_midi_vectors()
    x_lyrics_train, x_melody_train, x_lyrics_val, x_melody_val, y_train, y_val = dp.split_train_validation(x, one_hot_y,
                                                                                                           instance_to_song,
                                                                                                           midi_data_dict)
    embadding = pkl.load(open('emb_w.pkl', 'rb'))
    lyrics_model = m.build_model_2(embadding)
    rlop = ReduceLROnPlateau(min_delta=0.01)
    mcp = ModelCheckpoint('model2_checkpoint.h5', save_best_only=True)
    print('start fit')
    history = lyrics_model.fit([np.array(x_lyrics_train), np.array(x_melody_train)], np.array(y_train), epochs=5,
                               batch_size=512,
                               validation_data=([np.array(x_lyrics_val), np.array(x_melody_val)], np.array(y_val)),
                               verbose=1, callbacks=[rlop, mcp])
    lyrics_model.save('model_2.h5')
    return history, word_indexer


def generate_word(lyrics_gen_model, prev_word, midi_vec, word_idx):
    """
    generate next word for song. the choose is not determenistic.
    :param lyrics_gen_model: model that generate words
    :param prev_word: the previous word in the song
    :param midi_vec: the midi representation of the song
    :param word_idx: dictionary of all words.
    :return:
    """
    word_emb = np.array([word_idx[prev_word]])
    pred_words = lyrics_gen_model.predict([[word_emb], [midi_vec]])
    # the next lins choose word non determenistic. the probability is depend on the model's prediction
    acc_pred_vec = np.add.accumulate(pred_words, axis=1)
    rnd = np.random.random_sample()
    l_acc = len(acc_pred_vec)
    l_rnd = len(acc_pred_vec[acc_pred_vec > rnd])
    predicted_word = list(word_idx.keys())[l_acc - l_rnd - 1]
    return predicted_word


def train_autoencoder():
    """
    train autoencoder model
    :return:
    """
    midi_data_dict = pkl.load(open('midi_vectors.pkl', 'rb'))
    x_train = []
    x_val = []
    for key in midi_data_dict.keys():
        if np.random.random_sample() <= 0.2:
            x_val.append(midi_data_dict[key])
        else:
            x_train.append(midi_data_dict[key])
    x_train = np.array(x_train)
    x_val = np.array(x_val)
    ae_model = m.midi_autoencoder()
    ae_model[0].fit(x_train, x_train, validation_data=(x_val, x_val), epochs=20, batch_size=1)
    new_midi_dict = {}
    for key in midi_data_dict.keys():
        new_midi_dict[key] = ae_model[1].predict(np.array([midi_data_dict[key]]))[0]
    pkl.dump(new_midi_dict, open('encoded_midi_vectors', 'wb'))
    ae_model[1].save('autoencoder.h5')


def generate_test(model, word_indexer, song_length, test_path, use_autoencoder=False):
    """
    This function generate lyrics for all test songs
    :param model: the model that predict the lyrics
    :param word_indexer: dictionary for all words
    :param song_length: how many words to generate
    :param test_path: path of test file
    :param use_autoencoder: if use the shrink representation of midi vector
    :return:
    """
    autoencoder = load_model('autoencoder.h5')
    initial_words = ['hello', 'i', 'always']
    test_df = pd.read_csv(test_path, header=None)
    song_names = []
    for idx, row in test_df.iterrows():
        song_names.append(row[1])
        midi_vector = dp.midi_representation(os.path.join('midi_files', (str(row[0]) + ' - ' + str(row[1]) + '.mid')).replace(' ', '_'))
        if use_autoencoder:
            midi_vector = autoencoder.predict(np.array([midi_vector]))[0]
        print(f'song - {row[1]}')
        for word in initial_words:
            lyrics = []
            curr_word = word
            for i in range(song_length):
                curr_word = generate_word(model, curr_word, midi_vector, word_indexer)
                lyrics.append(curr_word)
            print(f'start word - {word}')
            print(' '.join(lyrics).replace('eos','\n'))
        print('******************************')
    print(song_names)



if __name__ == '__main__':
    # train_model_1()
    # train_model_2()
    # train_autoencoder()
    model = load_model('model2_checkpoint.h5')
    vocabulary_size = dp.get_vocabulary_size()
    x, one_hot_y, instance_to_song, song_indexes, word_indexer = dp.prepare_set(vocabulary_size)
    generate_test(model, word_indexer, 50, r"C:\Users\micha\Documents\michael\שנה ד\סמסטר ח\למידה עמוקה\Assignment3\lyrics_test_set.csv", True)
