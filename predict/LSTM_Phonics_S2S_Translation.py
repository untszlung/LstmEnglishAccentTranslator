#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 14:22:32 2021

@author: tszlung
"""
import numpy as np
from tensorflow import keras
import re
import json


batch_size = 64  # Batch size for training.
epochs = 500  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.

with open('../data/cmusphinxdict.json') as f:
  phoneme_dictionary = json.load(f)

#cmu phoneme list
cmu_phoneme = [
            'SPACE',
            'AA',
            'AE',
            'AH',
            'AO',
            'AW',
            'AY',
            'B',
            'CH',
            'D',
            'DH',
            'EH',
            'ER',
            'EY',
            'F',
            'G',
            'HH',
            'IH',
            'IY',
            'JH',
            'K',
            'L',
            'M',
            'N',
            'NG',
            'OW',
            'OY',
            'P',
            'R',
            'S',
            'SH',
            'T',
            'TH',
            'UH',
            'UW',
            'V',
            'W',
            'Y',
            'Z',
            'ZH',
        ] 

input_phonemes =['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z']
target_phonemes = ['\t', '\n', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z']
num_encoder_tokens = 38
num_decoder_tokens = 40
max_encoder_seq_length =  10
max_decoder_seq_length =  12
 
input_token_index = dict([(char, i) for i, char in enumerate(input_phonemes)])
target_token_index = dict([(char, i) for i, char in enumerate(target_phonemes)])



def word_to_phonics(input_word):
    phonics_blending = []
    for word in phoneme_dictionary:
        if (input_word == word):
#             print(input_word)
            for phoneme in phoneme_dictionary[word]:
                phoneme = re.split(' ',phoneme)
                phonics_blending.append(phoneme)
       
    return phonics_blending

def get_phoneme_index(input_phoneme):
    phoneme_index = 0
    
    for phoneme in cmu_phoneme:
        if (phoneme.upper() == input_phoneme):
            break

        phoneme_index += 1
    
    return phoneme_index

# Define sampling models
# Restore the model and construct the encoder and decoder.
model = keras.models.load_model("../model/phonics2phonics")

encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = keras.Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index["\t"]] = 1.0

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
#         if (sampled_char !='\t' or sampled_char != '\n'):
        decoded_sentence += sampled_char + ' '

        # Exit condition: either hit max length
        # or find stop character.
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.0

        # Update states
        states_value = [h, c]
    return decoded_sentence

while(True):
    i=0
    input_data = input("Enter your word: ")
    
    phoneme_blendings = word_to_phonics(input_data.upper())
    
    for phoneme_blending in phoneme_blendings:
 

 
    #     for phonics_blending in word_to_phonics(input_data.upper()):
        encoder_input_data_1 = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype="float32")
    #     for t, char in enumerate(input_data.upper()):

        input_word_phoneme = ''
        for t in range(len(phoneme_blending)):
           
            char = phoneme_blending[t]
            input_word_phoneme += char + ' '
    #         print(char)
    #         print(t)
            encoder_input_data_1[i, t, input_token_index[char]] = 1.0
    #     encoder_input_data_1[i, t + 1 :, input_token_index[" "]] = 1.0

        # print(encoder_input_data_1[0,1])
        input_seq = encoder_input_data_1[0 :  1]
        # print(input_seq)
        decoded_sentence = decode_sequence(input_seq)
        # decoded_sentence = decode_sequence(encoder_input_data_1[0:1])
        # print("-")
        #print("Input sentence:", input_texts[seq_index])
    #     decoded_output = re.split('\t|, |\n',decoded_sentence)

    #     decoded_sentence = ''
    #     for ouput in decoded_output:
    #         if output != '':
    #             decoded_sentence = decoded_sentence + output
        print("Input word phonics blending:", input_word_phoneme)
        print("Decoded word phonics blending:", decoded_sentence)
