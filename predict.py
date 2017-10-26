"""
Create a text sequence in Galician by continuing with the text entered in the command line.
We use a GRU network and weigths learning using Beiras texts.

Library needed: keras and tensorflow
Files needed :
    model_weights/best_beiras_gru_textdata_weights.hdf5 .- Network weights
    dictionaries.pkl .- Dictionaries to convert char to int and int to char using in learning.
"""

import sys
from beiras_aux import load_coded_dictionaries, predict_next_chars, clean_text
from keras.layers import Dense, Activation, GRU
from keras.models import Sequential

# Input size of the network, the entry text must have the same length
window_size = 100


def create_gru_model(number_chars):
    """
    Define the network
    :param
        numbers_chars .- Number chars using in the training process
    :return:
        model .- Model network defined
    """
    num_chars = len(number_chars)
    model = Sequential()
    # 1 Layer .- GRU layer 1 should be an GRU module with 200 hidden units
    model.add(GRU(200, input_shape=(window_size, num_chars), return_sequences=True))
    # 2 Layer .- GRU layer 2 should be an GRU module with 200 hidden units
    model.add(GRU(200))
    # 2 Layer .-  Dense, with number chars unit and softmax activation
    model.add(Dense(num_chars, activation='softmax'))
    return model


def predict(sentence):
    """
    Return a text sequence predicted by the GRU network continuing the input sentence
    :param
        sentence: Input sentence
    :return:
        text sequence
    """
    chars_to_indices, indices_to_chars = load_coded_dictionaries()
    model = create_gru_model(chars_to_indices)
    model.load_weights('model_weights/best_beiras_gru_textdata_weights.hdf5')
    return predict_next_chars(model, sentence, window_size, chars_to_indices, indices_to_chars)


if __name__ == "__main__":
    """
    Create a text sequence in Galician by continuing with the text entered in the command line.
    Input text must have at least window_size charazters (we only use window_size charazters).
    """
    # Read the input sentence
    input_sentence = ' '.join(sys.argv[1:])
    input_sentence = clean_text(input_sentence.lower())

    # Control input sentence len
    if len(input_sentence) < window_size:
        print("Sentence must have ", window_size, len(input_sentence))
        sys.exit(0)
    input_sentence = input_sentence[:window_size]
    # Predict
    predicted = predict(input_sentence)
    # Print
    print(input_sentence, "...", predicted)
