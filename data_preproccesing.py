import pandas as pd
import numpy as np
import re
import pretty_midi
import os
import nltk
from gensim.models import KeyedVectors
import time

nltk.download('punkt')
from keras.preprocessing.text import Tokenizer
import pickle as pkl


def lyrics_cleaning(lyrics):
    """
    clean lyrics text - lower capital, multi spaces and sub & with eos.
    :param lyrics: song lyrics
    :return: clean song lyrics
    """
    clean = lyrics.replace('&', 'eos')
    clean = clean.lower()
    clean = re.sub(' +', ' ', clean)
    return clean


def midi_representation(midi_file_path):
    """
    convert midi object in midi_file_path, to representation vector of the
    :param midi_file_path: midi file path
    :return: 297 length vector that represent the midi file
    """
    midi_object = pretty_midi.PrettyMIDI(midi_file_path)
    total_velocity = sum(sum(midi_object.get_chroma()))
    semitone = [sum(semitone) / total_velocity for semitone in midi_object.get_chroma()]
    piano_roll_norm = midi_object.get_piano_roll().sum(axis=1) / midi_object.get_piano_roll().sum()
    transition_matrix_norm = midi_object.get_pitch_class_transition_matrix(normalize=True).flatten()
    pc_histogram = midi_object.get_pitch_class_histogram()
    global_tempo_norm = np.array([midi_object.estimate_tempo() / 300])
    full_vector = np.concatenate((semitone, piano_roll_norm, transition_matrix_norm, pc_histogram, global_tempo_norm))
    full_vector[np.isnan(full_vector)] = 0
    return full_vector


def get_song_data(set_name):
    if set_name == 'train':
        songs_df = pd.read_csv(r'lyrics_train_set.csv', header=None)
    elif set_name == 'test':
        songs_df = pd.read_csv(r'lyrics_test_set.csv', header=None)

    songs_data = []
    for idx, row in songs_df.iterrows():
        mid_path = os.path.join('midi_files', (str(row[0]) + ' - ' + str(row[1]) + '.mid')).replace(' ', '_')
        midi_rep = midi_representation(mid_path)
        clean_lyrics = lyrics_cleaning(row[2])
        songs_data.append([clean_lyrics, midi_rep])

    return songs_data


def get_vocabulary_size():
    songs_df = pd.read_csv(r'lyrics_train_set.csv', header=None)
    vocabulary_set = set()
    for idx, row in songs_df.iterrows():
        clean = lyrics_cleaning(row[2])
        nltk_tokens = nltk.word_tokenize(clean)
        vocabulary_set |= set(nltk_tokens)
    return len(vocabulary_set)


def prepare_set(vocabulary_size):
    clean_lyrics = []
    song_indexes = []
    songs_df = pd.read_csv(r'lyrics_train_set.csv', header=None)
    for idx, row in songs_df.iterrows():
        song_indexes.append(
            [idx, os.path.join('midi_files', (str(row[0]) + ' - ' + str(row[1]) + '.mid')).replace(' ', '_')])
        clean_lyrics.append(lyrics_cleaning(row[2]))
    tok = Tokenizer(num_words=vocabulary_size)
    tok.fit_on_texts(clean_lyrics)
    instances = tok.texts_to_sequences(clean_lyrics)
    word_indexer = tok.word_index
    instance_to_song = []
    instances_list = []
    for idx, instance in enumerate(instances):
        for i in range(len(instance) - 1):
            instances_list.append(instance[i:i + 2])
            instance_to_song.append(idx)

    x = np.array(instances_list)[:, 0]
    y = np.array(instances_list)[:, 1]
    one_hot_y = np.zeros((y.size, len(word_indexer) + 1))
    one_hot_y[np.arange(y.size), y] = 1
    return x, one_hot_y, instance_to_song, song_indexes, word_indexer


def prepare_embedding(word_idx):  # TODO: check with new wiki model download
    word2vec = KeyedVectors.load_word2vec_format(r'w2v_model\fasttextmodel.vec')
    emb_w = np.zeros((len(word_idx) + 1, 300))
    for w, index in word_idx.items():
        emb_w[index, :] = word2vec[w] if w in word2vec else np.zeros(300)
    return emb_w


def get_midi_vectors():
    midi_data = {}
    songs_df = pd.read_csv(r'lyrics_train_set.csv', header=None)
    for idx, row in songs_df.iterrows():
        time1 = time.time()
        try:
            midi_data[idx] = midi_representation(
                os.path.join('midi_files', (str(row[0]) + ' - ' + str(row[1]) + '.mid')).replace(' ', '_'))
        except:
            pass
        time2 = time.time()
        print('%s song : function took %0.3f ms' % (idx,(time2 - time1) * 1000.0))
    pkl.dump(midi_data,open('midi_vectors.pkl', 'wb'))
    return midi_data


def split_train_validation(X, y, instance_to_song, midi_data):
    x_lyrics_train = []
    x_melody_train = []
    x_lyrics_val = []
    x_melody_val = []
    y_train = []
    y_val = []

    for i in range(X.shape[0]):
        if np.random.random_sample() <= 0.2:
            try:
                x_melody_val.append(midi_data[instance_to_song[i]])
            except:
                continue
            x_lyrics_val.append(X[i])
            y_val.append(y[i])
        else:
            try:
                x_melody_train.append(midi_data[instance_to_song[i]])
            except:
                continue
            x_lyrics_train.append(X[i])
            y_train.append(y[i])
    return x_lyrics_train, x_melody_train, x_lyrics_val, x_melody_val, y_train, y_val

y = midi_representation('midi_files/38_Special_-_Caught_Up_In_You.mid')
# get_song_data('train')
# vec_size = get_vocabulary_size()
# x,y,instance_to_song,song_indexes,word_indexer = prepare_set(vec_size)
# prepare_embedding(word_indexer)
# emw = pkl.load(open('emb_w.pkl','rb'))
x=0
