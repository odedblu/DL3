import models as m
import pickle as pkl
import data_preproccesing as dp
import numpy as np
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from keras.models import load_model


def train_model_1():
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


# train_model_1()


def generate_word(lyrics_gen_model, prev_word, midi_vec, word_idx):
    word_emb = np.array([word_idx[prev_word]])
    pred_words = lyrics_gen_model.predict([[word_emb], [midi_vec]])
    acc_pred_vec = np.add.accumulate(pred_words, axis=1)
    rnd = np.random.random_sample()
    l_acc = len(acc_pred_vec)
    l_rnd = len(acc_pred_vec[acc_pred_vec > rnd])
    predicted_word = list(word_idx.keys())[l_acc - l_rnd]
    return predicted_word


def generate_song(lyrics_gen_model, start_word, midi_vec, word_idx, length):
    lyrics = [start_word]
    curr_word = start_word
    for i in range(length):
        curr_word = generate_word(lyrics_gen_model, curr_word, midi_vec, word_idx)
        lyrics.append(curr_word)
    return lyrics

model = load_model('model_checkpoint.h5')
midi_test_vec = dp.midi_representation(r'midi_files/Aqua_-_Barbie_Girl.mid')
vocabulary_size = dp.get_vocabulary_size()
x, one_hot_y, instance_to_song, song_indexes, word_indexer = dp.prepare_set(vocabulary_size)
# generate_word(model, 'hi', midi_test_vec,word_indexer)
song_ly = generate_song(model, 'hi', midi_test_vec,word_indexer,50)
print(song_ly)
