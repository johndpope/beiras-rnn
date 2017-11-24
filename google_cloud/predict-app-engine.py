# -*- coding: utf-8 -*-
"""
Create a text sequence in Galician by continuing with the text
entered in the command line.
We use a GRU model serving by google-cloud-ml

Library needed: tensorflow_serving,grpc and numpy
Files needed :
    model_weights/best_beiras_gru_textdata_weights.hdf5 .- Network weights
    dictionaries_0.pkl .- Dictionaries to convert char to int and int to char
                         using in learning.
"""

import sys
import requests
import json
sys.path.insert(0, '../aux/')
from beiras_aux import clean_text


# Input size of the network, the entry text must have the same length
window_size = 100


# Complete a sequence using the server
def predict_window(text_predict, number_predict, window_size):
    # Get dictionaries
    input_clean = text_predict
    # Call server for all charazters

    url = "http://ai-ml-dl.appspot.com/api/beiras_rnn"
    # url = "http://localhost:8080/api/beiras_rnn"
    data = {"input": text_predict}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(url, data=json.dumps(data), headers=headers)
    if r.status_code == requests.codes.ok:
        return r.json()["output"]
    else:
        if r.status_code == 400:
            error_code = r.json()["error_code"]
            if error_code == 2:
                return ("Error in data lenght. Minimum length = 100")
            return ("Error in data format")
        if r.status_code == 500:
            return ("Error in server")
        else:
            return("Error " + str(r.json()))
    return ""


def predict(sentence, number_predict, window_size):
    """
    Return a text sequence predicted by the GRU network continuing
    the input sentence
    :param
        sentence: Input sentence
    :return:
        text sequence
    """
    return predict_window(sentence, number_predict, window_size)


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
    input_sentence = input_sentence[:window_size]
    # Predict
    predicted = predict(input_sentence, window_size, window_size)
    # Print
    print(predicted)
