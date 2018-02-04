# -*- coding: utf-8 -*-
"""
Auxilar funcions for beiras_rnn
There is 3 group of function
- Clean and load the text file
- Create a text sequeence.
- Save and load dictionaries
- A main part to test dictionaries part
"""

import sys
import numpy as np
import pandas as pd
import re
import pickle
import csv
import tensorflow as tf
sys.path.insert(0, '../aux/')
from beiras_aux import save_coded_dictionaries,load_coded_dictionaries, \
    predict_next_chars, clean_text,load_text, encode_io_pairs,\
    window_transform_text
import codecs





WINDOW_SIZE = 100
STEP_SIZE = 1
FILE_OUTPUT_TRAIN= "data/beiras_train.csv"
FILE_OUTPUT_TEST= "data/beiras_eval.csv"
PERCENT_TRAIN=0.8

def load_text_clean(sz_file, l_char_to_index):
    """
    Load a text, clean
    :param sz_file : path to input file, a text file
    :param l_window_size= int size of text window to use
    :param l_step_size
    :param l_char_to_index dictionaty to convert char to integer
    :return: String with the text clean
    """
    l_text_org = codecs.open(sz_file, encoding="utf-8").read().lower()
    l_text_clean = clean_text(l_text_org)
    change=True
    while change:
        change=False;
        chars_text = sorted(list(set(l_text_clean)))
        for i,char in enumerate(chars_text):
            if not change:
                if char not in l_char_to_index.keys():
                    l_text_clean=l_text_clean.replace(char,'')
                    change=True
    return l_text_clean


def text_array_to_csv(text_array,csv_file,l_char_to_index):
    """
    Save a array to csv file
    :param text_array: array text to save
    :param csv_file: file to generate
    :param l_char_to_index: dictionaty to convert char to index
    :return: None
    """
    with open(csv_file, 'w') as file:
        writer = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for sentence in text_array:
            X = np.zeros(WINDOW_SIZE +1,dtype=int)
            for t, char in enumerate(sentence):
                X[t]=int(l_char_to_index[char])
            writer.writerow(X)

def text_to_csv(text,csv_file_train,csv_file_test,l_char_to_index,percent_test):
    """
    From a text, generate the csv_file_train,csv_file_test
    :param text: Text to convert
    :param csv_file_train:  File for train
    :param csv_file_test:  File for text
    :param l_char_to_index: dictionaty to convert char to index
    :param percent_test:  Percent to use as text
    :return:  None
    """

    # inputs:List of sentences of size WINDOW_SIZE+1
    # outputs: List of chars, being the char ith the next char to sentence inputs[ith]
    inputs, outputs = window_transform_text(text, WINDOW_SIZE + 1, STEP_SIZE)
    train_size=int(percent_test * len(inputs))

    text_array_to_csv(inputs[:train_size], csv_file_train, l_char_to_index)
    text_array_to_csv(inputs[train_size:], csv_file_test, l_char_to_index)


def csv_to_text(csv_file,index_to_char):
    text=[]
    with open(csv_file, 'r') as file:
        reader = csv.reader(file, delimiter=',', quotechar='|')
        for row in reader:
            sentence=""
            for index in row:
                sentence= sentence +  index_to_char[int(index)]
            text.append(sentence)
    return text




if __name__ == "__main__":
    """
    Generate FILE_OUTPUT_TRAIN and FILE_OUTPUT_TEST from  '../data/Beiras.txt'
    """

    #Load dictionaries to convert char to index and index to char
    chars_to_indices_new, indices_to_chars_new = load_coded_dictionaries()

    # Get String with clean text
    text_clean = load_text_clean('../data/Beiras.txt',chars_to_indices_new )
    #Generate files
    text_to_csv(text_clean, FILE_OUTPUT_TRAIN,FILE_OUTPUT_TEST,chars_to_indices_new,PERCENT_TRAIN)


    # For test, read the file and test
    text_read=csv_to_text(FILE_OUTPUT_TRAIN, indices_to_chars_new)
    for sentence in text_read:
        print(sentence)



