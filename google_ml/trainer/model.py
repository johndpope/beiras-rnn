
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
    model = Sequential()
    # 1 Layer .- GRU layer 1 should be an GRU module with 200 hidden units
    model.add(GRU(
                    200, input_shape=(window_size, number_chars),
                    return_sequences=True
                )
            )
    # 2 Layer .- GRU layer 2 should be an GRU module with 200 hidden units
    model.add(GRU(200))
    # 2 Layer .-  Dense, with number chars unit and softmax activation
    model.add(Dense(number_chars, activation='softmax'))
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

  #TensorFlow serving expects you to point to a base directory which includes a version subdirectory.
  # We create it when we are in local
  if not export_path.startswith("gs://"):
      export_path=export_path + "/2"
  builder = saved_model_builder.SavedModelBuilder(export_path)

  signature = predict_signature_def(inputs={'input': model.inputs[0]},
                                    outputs={'sequence': model.outputs[0]})

  with K.get_session() as sess:
    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature}
    )
    builder.save()


def input_to_matrix(inputs,num_chars,col_label):
    label = inputs.pop(col_label)
    x_matrix = np.zeros((inputs.shape[0], inputs.shape[1], num_chars), dtype=np.bool)
    y_matrix = np.zeros((len(inputs), num_chars), dtype=np.bool)

    for i, sentence in enumerate(inputs):
        for t, char in enumerate(sentence):
            x_matrix[i, t, int(char)] = 1
        y_matrix[i, int(label[i])] = 1
    return x_matrix,y_matrix



def generator_input(input_file, chunk_size,window_size,num_chars):
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
    n_rows = input_data.shape[0]
    x,y=input_to_matrix(input_data,num_chars,col[window_size])
    #GRU in keras need to have a shape N,window_len,input.shape
    return ((np.reshape(x[index % n_rows],(1,x.shape[1],x.shape[2])), np.reshape(y[index % n_rows],(1,y.shape[1]))) for index in
            itertools.count())


