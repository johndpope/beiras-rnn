# -*- coding: utf-8 -*-
"""
Create a text sequence in Galician by continuing with the text entered
in the command line.
We use a GRU model serving by google-cloud-ml

Library needed: tensorflow_serving,grpc and numpy
Files needed :
    model_weights/best_beiras_gru_textdata_weights.hdf5 .- Network weights
    dictionaries_0.pkl .- Dictionaries to convert char to int and int to char
    using in trainig.
"""

import sys
import tensorflow as tf
import grpc
import numpy as np
import googleapiclient.discovery
sys.path.insert(0, '../aux/')
from beiras_aux import load_coded_dictionaries, predict_next_chars, clean_text


# Input size of the network, the entry text must have the same length
window_size = 100
project = "ai-ml-dl"
model = "BeirasRnn"
version = "v10"


# Define a function
def predict_one(
                text_predict, service, model_name, window_size,
                chars_to_indices, indices_to_chars
                ):
    # Convert input sequence to array
    number_chars = len(chars_to_indices)
    x_test = np.zeros((window_size, number_chars))
    for t, char in enumerate(text_predict):
        x_test[t, chars_to_indices[char]] = 1.
    x_test = x_test[:window_size, :]
    # Prepare the request
    instances = {'sequence': x_test.tolist()}

    response = service.projects().predict(
        name=model_name,
        body={'instances': instances}
    ).execute()
    if 'error' in response:
        raise RuntimeError(response['error'])
    test_predict = np.array(response['predictions'][0]['scores'])
    r = np.argmax(test_predict)  # predict class of each test input
    return (indices_to_chars[r])


# Complete a sequence using the server
def predict_window(
        text_predict, number_predict, window_size, lproject, lmodel, version
        ):
    # Get dictionaries
    chars_to_indices, indices_to_chars = load_coded_dictionaries()
    # Get stub
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(lproject, lmodel)
    if lversion is not None:
        name += '/versions/{}'.format(lversion)
    print(name)
    input_clean = text_predict
    # Call server for all charazters
    for i in range(number_predict):
        d = predict_one(
            input_clean[i:], service, name, window_size,
            chars_to_indices, indices_to_chars
        )
        input_clean += d
    return input_clean


def predict(sentence, number_predict, window_size):
    """
    Return a text sequence predicted by the GRU network
    continuing the input sentence
    :param
        sentence: Input sentence
    :return:
        text sequence
    """
    return predict_window(
                sentence, number_predict, window_size,
                project, model, version
            )


if __name__ == "__main__":
    """
    Create a text sequence in Galician by continuing with the text entered
    in the command line.
    Input text must have at least window_size charazters
    (we only use window_size charazters).
    """
    # Read the input sentence
    input_sentence = ' '.join(sys.argv[1:]).decode("utf-8")
    input_sentence = clean_text(input_sentence.lower())

    # Control input sentence len
    if len(input_sentence) < window_size:
        print("Sentence must have ", window_size, len(input_sentence))
        sys.exit(0)
    input_sentence = input_sentence[:window_size]
    # Predict
    predicted = predict(input_sentence, window_size, window_size)
    # Print
    print(predicted)
