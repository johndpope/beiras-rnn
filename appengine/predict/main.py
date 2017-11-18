# -*- coding: utf-8 -*-
# Copyright 2016 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from google.appengine.ext import vendor
# Add any libraries installed in the "lib" folder.
vendor.add('lib')

import webapp2
import json
import logging
import grpc
import numpy as np
import googleapiclient.discovery
import re
import pickle

window_size = 100
predict_size=100
project="ai-ml-dl"
model="BeirasRnn"
version="v10"

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


def load_coded_dictionaries():
    with open('./dictionaries_0.pkl', 'rb') as output:
        l_chars_to_indices = pickle.load(output)
        l_indices_to_chars = pickle.load(output)
    return l_chars_to_indices, l_indices_to_chars


# Define a function
def predict_one(text_predict,service,model_name,window_size,chars_to_indices, indices_to_chars):
    # Convert input sequence to array
    number_chars=len(chars_to_indices)
    x_test = np.zeros((window_size,number_chars))
    for t, char in enumerate(text_predict):
       x_test[t,chars_to_indices[char]] = 1.
    #print(x_test.shape)
    x_test=x_test[:window_size,:]
    
    #Prepare the request
    instances={'sequence':x_test.tolist()}
    response = service.projects().predict(
        name=model_name,
        body={'instances': instances}
    ).execute()
    if 'error' in response:
        raise RuntimeError(response['error'])
    test_predict=np.array(response['predictions'][0]['scores'])
    r = np.argmax(test_predict)  # predict class of each test input
    logging.debug(str(test_predict) + " " + str(r) )
    return (indices_to_chars[r])

# Complete a sequence using the server
def predict_window(text_predict,number_predict,window_size,lproject,lmodel,lversion):
    
    # Get dictionaries
    chars_to_indices, indices_to_chars = load_coded_dictionaries()
    # Get stub
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(lproject, lmodel)
    if lversion is not None:
        name += '/versions/{}'.format(lversion)
    print(name)
    input_clean=text_predict;
    # Call server for all charazters
    for i in range(number_predict):
        logging.debug("predict_window [" +  input_clean + "]")
        d=predict_one(input_clean[i:],service,name,window_size,chars_to_indices, indices_to_chars)
        input_clean+=d
    return input_clean

def predict(sentence,number_predict,window_size):
    """
    Return a text sequence predicted by the GRU network continuing the input sentence
    :param
        sentence: Input sentence
    :return:
        text sequence
    """
    return predict_window(sentence,number_predict,window_size,project,model,version)


class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers.add_header('Access-Control-Allow-Origin', '*')
        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Hiola, World!')

class BeirasRnn(webapp2.RequestHandler):
    def post(self):
        self.response.headers.add_header('Access-Control-Allow-Origin', '*')
        json_request=json.loads(self.request.body)
        input_string = json_request["input"]
        input_string =clean_text(input_string.lower())
        logging.debug("Log 3 " +  input_string )
       
        if (len (input_string)<window_size):
            self.response.headers['Content-Type'] =  'application/json'	
            self.response.write(json.dumps({"retorno" : "" , "error" : "No_lenght"}))
            return
        logging.debug("Log 4 " +  input_string[:window_size] )
        return_string=predict(input_string[:window_size],predict_size,window_size)
        self.response.headers['Content-Type'] =  'application/json'	
        self.response.write(json.dumps({"output" : return_string}))



app = webapp2.WSGIApplication([
    ('/', MainPage),('/api/beiras_rnn',BeirasRnn)
], debug=True)
