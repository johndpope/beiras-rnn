
import itertools

import keras
import pandas as pd
import numpy as np
from keras import backend as K
from keras import layers, models
from keras.utils import np_utils
from keras.backend import relu, sigmoid
from keras.layers import Dense, Activation, GRU
from keras.models import Sequential

from urlparse import urlparse

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model import tag_constants, signature_constants
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def, predict_signature_def
from tensorflow.contrib.session_bundle import exporter


def model_fn(number_chars,learning_rate=0.001,window_size = 100):
    num_chars = len(number_chars)
    model = Sequential()
    # 1 Layer .- GRU layer 1 should be an GRU module with 200 hidden units
    model.add(GRU(
                    200, input_shape=(window_size, num_chars),
                    return_sequences=True
                )
            )
    # 2 Layer .- GRU layer 2 should be an GRU module with 200 hidden units
    model.add(GRU(200))
    # 2 Layer .-  Dense, with number chars unit and softmax activation
    model.add(Dense(num_chars, activation='softmax'))
    compile_model(model, learning_rate)
    return model


def compile_model(model, learning_rate):
    optimizer = keras.optimizers.RMSprop(lr=learning_rate, rho=0.9, epsilon=1e-08, decay=0.0)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy'])
    return model


def to_savedmodel(model, export_path):
  """Convert the Keras HDF5 model into TensorFlow SavedModel."""

  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                    outputs={'income': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()


def generator_input(input_file, chunk_size,window_size):
  """Generator function to produce features and labels
     needed by keras fit_generator.
  """
  col=[]
  for k in range(0,window_size+1):
    col.append(str(k))
  input_reader = pd.read_csv(tf.gfile.Open(input_file[0]),
                           chunksize=chunk_size,
                             names=col)

  for input_data in input_reader:
    label = input_data.pop(col[window_size])
    n_rows = input_data.shape[0]
    return ( (input_data.iloc[[index % n_rows]], label.iloc[[index % n_rows]]) for index in itertools.count() )
