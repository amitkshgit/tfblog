# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf
import os
import json
import pickle
import sys
import signal
import traceback
import flask
import pandas as pd
import urllib


prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph


def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result


def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label



class ScoringService(object):
    graph = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        model_file = "/opt/ml/model/output_graph.pb"
        if cls.graph == None:
           cls.graph = load_graph(model_file)
        return cls.graph

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        input_height = 299
        input_width = 299
        input_mean = 128
        input_std = 128
        input_layer = "Mul"
        output_layer = "final_result"
        #file_name = "/opt/ml/input/data/train/labelthis.jpg"
        file_name = "/tmp/labelthis.jpg"
        clf = cls.get_model()
        print("Input: " + input)
        urllib.urlretrieve (input,file_name)
        t = read_tensor_from_image_file(
      		file_name,
      		input_height=input_height,
      		input_width=input_width,
      		input_mean=input_mean,
      		input_std=input_std)

        input_name = "import/" + input_layer
        output_name = "import/" + output_layer
        #input_operation = graph.get_operation_by_name(input_name)
        #output_operation = graph.get_operation_by_name(output_name)
        input_operation = clf.get_operation_by_name(input_name)
        output_operation = clf.get_operation_by_name(output_name)

        with tf.Session(graph=clf) as sess:
                results = sess.run(output_operation.outputs[0], {
                input_operation.outputs[0]: t
        })
        results = np.squeeze(results)

        label_file = "/opt/ml/model/output_labels.txt"
        top_k = results.argsort()[-5:][::-1]
        labels = load_labels(label_file)
        pred = ""
        jsonPred = {}
        for i in top_k:
                print(labels[i], results[i])
                pred = pred + labels[i] + "," + str(results[i]) + "\n"
                jsonPred[labels[i]] = str(results[i])

        jsonPredData = json.dumps(jsonPred)
        #pred = labels[0] + "," + str(results[0])

        return jsonPredData


# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'text/plain':
	print("In trans 1: " + flask.request.data)
        data = flask.request.data.decode('utf-8')
        s = data
	print("In trans 2: " + s)
        #data = pd.read_csv(s, header=None)
    else:
        return flask.Response(response='This predictor only supports TXT data', status=415, mimetype='text/plain')

    print('Invoked with {} records'.format(data))

    # Do the prediction
    predictions = ScoringService.predict(data)

    # Convert from numpy back to CSV
    #out = StringIO.StringIO()
    #pd.DataFrame({'results':predictions}).to_csv(out, header=False, index=False)
    #result = out.getvalue()

    return flask.Response(response=predictions, status=200, mimetype='text/csv')




