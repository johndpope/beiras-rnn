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

import webapp2
import json
import logging
from beiras_aux import load_coded_dictionaries, predict_next_chars, clean_text
import tensorflow as tf
import grpc
import numpy as np
import googleapiclient.discovery

class MainPage(webapp2.RequestHandler):
    def get(self):
        self.response.headers['Content-Type'] = 'text/plain'
        self.response.write('Hiola, World!')

class BeirasRnn(webapp2.RequestHandler):
    def post(self):

        logging.debug("Log 1 " + str(self.request.POST))
        json_request=json.loads(self.request.body)
        logging.debug("Log 2 " +str(json_request))
        logging.debug("Log 3 " + str(json_request["input"]))
        input_string = json_request["input"]
        self.response.headers['Content-Type'] =  'application/json'	
        self.response.write(json.dumps({"retorno" : "Probando retorno 2 : " + input_string}))


app = webapp2.WSGIApplication([
    ('/', MainPage),('/api/beiras_rnn',BeirasRnn)
], debug=True)
