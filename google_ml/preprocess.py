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




CHUCK=1000
WINDOW_SIZE = 100
STEP_SIZE = 1
FILE_OUTPUT_TRAIN= "data/beiras_train.csv"
FILE_OUTPUT_TEST= "data/beiras_test.csv"
PERCENT_TRAIN=0.8

def load_text_clean(sz_file, l_window_size, l_step_size,l_char_to_index):
    """
    Load a text, clean it and windownize
    """
    l_text_org = codecs.open(sz_file, encoding="utf-8").read().lower()
    l_text_clean = clean_text(l_text_org)
    change=True
    while change:
        change=False;
        chars_text = sorted(list(set(l_text_clean)))
        chars_dict = sorted(list(set(l_char_to_index.keys())))
        for i,char in enumerate(chars_text):
            if not change:
                if char not in l_char_to_index.keys():
                    l_text_clean=l_text_clean.replace(char,'')
                    change=True
    return l_text_clean


def text_array_to_csv(text_array,csv_file,l_char_to_index):
    with open(csv_file, 'w') as file:
        writer = csv.writer(file, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for sentence in text_array:
            X = np.zeros(WINDOW_SIZE +1,dtype=int)
            for t, char in enumerate(sentence):
                X[t]=int(l_char_to_index[char])
            writer.writerow(X)

def text_to_csv(text,csv_file_train,csv_file_test,l_char_to_index,percent_test):
    inputs, outputs = window_transform_text(text, WINDOW_SIZE + 1, STEP_SIZE)
    train_size=int(percent_test * len(inputs))
    print("Train " + str(train_size))
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

def csv_to_pandas(csv_file):
    input_reader = pd.read_csv(tf.gfile.Open(csv_file),
                               chunksize=100)
    for input_data in input_reader:


if __name__ == "__main__":
    """
    Test the functions to  save and load dictionaries
    """


    chars_to_indices_new, indices_to_chars_new = load_coded_dictionaries()


    text_clean = load_text_clean('../data/Beiras.txt',WINDOW_SIZE +1 ,
                                STEP_SIZE,chars_to_indices_new )


    chars_text = sorted(list(set(text_clean)))
    chars_dict = sorted(list(set(chars_to_indices_new.keys())))


    text_to_csv(text_clean, FILE_OUTPUT_TRAIN,FILE_OUTPUT_TEST,chars_to_indices_new,PERCENT_TRAIN)

    text_read=csv_to_text(FILE_OUTPUT_TRAIN, indices_to_chars_new)
    for sentence in text_read:
        print(sentence)



