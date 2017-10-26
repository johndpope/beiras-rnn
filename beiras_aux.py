"""
Auxilar funcions for beiras_rnn
There is 3 group of function
- Clean and load the text file
- Create a text sequeence.
- Save and load dictionaries
- A main part to test dictionaries part
"""

import numpy as np
import re
import pickle

"""
Funtions Clean and load the text file
"""


def window_transform_text(text, l_window_size, l_step_size):
    """
    Use sliding window to extract input/output pairs from a text
    params:
        text .- text to windownize
        l_windows_size .- Len of any input sentence to return
    Return:
        input .- array of sentence 
        output .- for any sentence in input, the next charazter
    """
    # containers for input/output pairs
    inputs = []
    outputs = []
    # Number of windows to create
    n_windows = int((len(text) - l_window_size) / l_step_size)
    for j in range(n_windows):
        # k .- Start index
        k = j * l_step_size
        inputs.append(text[k:(k + l_window_size)])
        outputs.append(text[k + l_window_size])
    return inputs, outputs


def encode_io_pairs(text, l_window_size, l_step_size, l_chars_to_indices):
    """
     transform character-based input/output into equivalent numerical versions
    """
    # number of unique chars
    num_chars = len(sorted(list(set(text))))

    # cut up text into character input/output pairs
    inputs, outputs = window_transform_text(text, l_window_size, l_step_size)

    # create empty vessels for one-hot encoded input/output
    X_return = np.zeros((len(inputs), l_window_size, num_chars), dtype=np.bool)
    y_return = np.zeros((len(inputs), num_chars), dtype=np.bool)

    # loop over inputs/outputs and tranform and store in X_return/y_return
    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            X_return[i, t, l_chars_to_indices[char]] = 1
        y_return[i, l_chars_to_indices[outputs[i]]] = 1

    return X_return, y_return


def clean_text(l_text_org):
    """
    Clean a text
    """
    text_without_source = ""
    regexp = re.compile(r'http')
    for line in l_text_org.splitlines():
        if not regexp.search(line):
            text_without_source = text_without_source + line
    l_text_clean = re.sub('[ºªàâäçèêïìôöü&%@•…«»”“*/!"(),.:;_¿¡¿‘’´\[\]\']', ' ', text_without_source)
    l_text_clean = l_text_clean.replace("  ", " ")
    return l_text_clean


def load_text(sz_file, l_window_size, l_step_size):
    """
    Load a text, clean it and windownize
    """
    l_text_org = open(sz_file, encoding="utf-8").read().lower()
    l_text_clean = clean_text(l_text_org)
    l_chars = sorted(list(set(l_text_clean)))
    # this dictionary is a function mapping each unique character to a unique integer
    l_chars_to_indices = dict((c, i) for i, c in enumerate(l_chars))
    # this dictionary is a function mapping each unique integer back to a unique character
    l_indices_to_chars = dict((i, c) for i, c in enumerate(l_chars))
    X_return, y_return = encode_io_pairs(l_text_clean, l_window_size, l_step_size, l_chars_to_indices)
    return X_return, y_return, l_chars, l_chars_to_indices, l_indices_to_chars, l_text_clean


"""
Functions to create a text sequence.
"""


def predict_next_chars(model, input_chars, l_window_size, l_chars_to_indices, l_indices_to_chars):
    """
    Predict next l_window_size character from a sentence using a model l_chars_to_indices and indices to chars are
    dictionaries uses to translate char in index.    Must be the same that used in training the model.
    """
    # create output
    number_chars = len(l_chars_to_indices)
    predicted_chars = ''
    for i in range(l_window_size):
        # convert this round's predicted characters to numerical input    
        x_test = np.zeros((1, l_window_size, number_chars))
        for t, char in enumerate(input_chars):
            x_test[0, t, l_chars_to_indices[char]] = 1.

        # make this round's prediction
        test_predict = model.predict(x_test, verbose=0)[0]

        # translate numerical prediction back to characters
        r = np.argmax(test_predict)  # predict class of each test input
        d = l_indices_to_chars[r]

        # update predicted_chars and input
        predicted_chars += d
        input_chars += d
        input_chars = input_chars[1:]
    return predicted_chars


def print_predicctions(model, weights_file, l_chars_to_indices, l_indices_to_chars,
                       l_text_clean, l_window_size):
    """
    Print predicctions for sentence beginning in position 100,1000 and 5000 of l_text_clean l_chars_to_indices and
    indices to chars are dictionaries uses to translate char in index. Must be the same that used in training the
    model.
    """
    start_inds = [100, 1000, 5000]

    # load in weights
    model.load_weights(weights_file)
    for s in start_inds:
        start_index = s
        input_chars = l_text_clean[start_index: start_index + l_window_size]

        # use the prediction function
        predict_input = predict_next_chars(model, input_chars, l_window_size, l_chars_to_indices, l_indices_to_chars)

        print(input_chars + "...." + predict_input)


"""
Functions to  save and load dictionaries        
"""


def save_coded_dictionaries(l_chars_to_indices, l_indices_to_chars):
    with open('dictionaries.pkl', 'wb') as output:
        pickle.dump(l_chars_to_indices, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(l_indices_to_chars, output, pickle.HIGHEST_PROTOCOL)


def load_coded_dictionaries():
    with open('dictionaries.pkl', 'rb') as output:
        l_chars_to_indices = pickle.load(output)
        l_indices_to_chars = pickle.load(output)
    return l_chars_to_indices, l_indices_to_chars


def encode_text(text, l_chars_to_indices):
    """
    Encode a text using a dictionary
    """
    l_text_coded = np.zeros((len(text)))
    for t, char in enumerate(text):
        l_text_coded[t] = l_chars_to_indices[char]
    return l_text_coded


def decode_text(l_text_coded, l_indices_to_chars):
    """
        Encode a text using a dictionary
    """
    l_text_decoded = ""
    for t, index in enumerate(l_text_coded):
        l_text_decoded += l_indices_to_chars[index]
    return l_text_decoded


if __name__ == "__main__":
    """
    Test the functions to  save and load dictionaries  
    """
    window_size = 100
    step_size = 1
    X, y, chars, chars_to_indices, indices_to_chars, text_clean = load_text('Beiras.txt', window_size, step_size)
    save_coded_dictionaries(chars_to_indices, indices_to_chars)
    chars_to_indices_new, indices_to_chars_new = load_coded_dictionaries()
    text_org = text_clean[:100]
    print(text_org)
    text_coded = encode_text(text_org, chars_to_indices)
    print(text_coded)
    text_decoded = decode_text(text_coded, indices_to_chars_new)
    print(text_decoded)
