#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 17:22:16 2021

@author: tszlung
"""
from numpy import array
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re
import json

SEQUENCE_MAX_LENGTH =15
total_output_classes = 278



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

# create a sequence classification instance
def get_sequence(input_data_alphabet_sequence, output_data_list,index):

	# reshape input and output data to be suitable for LSTMs
	X = array(input_data_alphabet_sequence).reshape(len(input_data_alphabet_sequence), SEQUENCE_MAX_LENGTH,1)
	y = array(output_data_list).reshape(len(input_data_alphabet_sequence),total_output_classes)
    
	return X, y


#read training data
input_data_list = []
output_data_list = []
with open('../data/wordfun_phonics.txt') as f:
    lines = f.readlines()
    
count = 0    
for line in lines:
    count += 1
    #print(f'line {count}: {line}')    
    input_output = re.split('; |, |\n',line)
    #output = input_output[0]
    #print(output)
    wordCount =0
    output_word = ''
    for word in input_output:
        if wordCount == 0:
            output_word = word
        else:
            if word != '':
                input_data_list.append(word)
                output_data_list.append(output_word)
        
        wordCount += 1
  
label_encoder = LabelEncoder()
label_encoder.fit_transform(output_data_list)
        


#Load cmu dictionary 
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




 
    
 #load model
model = keras.models.load_model('../model/lstm_translator_phonics_model')

#predict the word
while(True):
    input_data = input("Enter your word: ")


    #print(input_data)
    for phonics_blending in word_to_phonics(input_data.upper()):
        print(f'-Input word phonics blending : {phonics_blending}')
        single_word_phoneme_blending_index = []
        for phoneme in phonics_blending:
            #print(get_phoneme_index(phoneme))
            single_word_phoneme_blending_index.append(get_phoneme_index(phoneme))

        #insert space after word
        #single_word_phoneme_blending_index.append(get_phoneme_index('SPACE'))

        #add ending padding
        end_padding = range(SEQUENCE_MAX_LENGTH-len(single_word_phoneme_blending_index))
        for end_padding_index in end_padding:
            #set the padding valoue to 100
            single_word_phoneme_blending_index.append(get_phoneme_index('SPACE'))

#     world_index = 14
#     temp = X[world_index].reshape(1,15,1)

        alphabet_sequence = array(single_word_phoneme_blending_index).reshape(15,1)
        #print(alphabet_sequence)
        temp = alphabet_sequence.reshape(1,15,1)

        #print(temp)

        output_accuracy = model.predict(temp)

        y_pred = np.argmax(output_accuracy, axis=1)

        predict_output = label_encoder.inverse_transform(array(y_pred))

        #print(input_data_list[world_index])
        #print(input_data)
        print('-Predicted word: ' + predict_output[0] + ' (' +  str(output_accuracy[0][y_pred][0]) + ')\n')
  