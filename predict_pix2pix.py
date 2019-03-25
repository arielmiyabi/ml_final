import tensorflow as tf
import numpy as np
import cv2 as cv
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
import os
import argparse

imgsize = 256

parser = argparse.ArgumentParser()

parser.add_argument("--input_sketch", dest='sketch',  nargs='?',
                    help="the sketch path you want to input.")

args = parser.parse_args()

def plot(sample):
    f_sample = np.asarray(sample).flatten()
    normalized = (f_sample-min(f_sample))/(max(f_sample)-min(f_sample))*255
    cv.imwrite('result/test/{}'.format(str(args.sketch.split('/')[-1]).zfill(3)), normalized.reshape(imgsize, imgsize))

os.makedirs('result/test', exist_ok=True)
img = cv.imread(args.sketch, cv.IMREAD_GRAYSCALE)/255
img = cv.resize(img, (imgsize, imgsize))
img = np.reshape(img, [1, imgsize, imgsize, 1])

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./model/model-/4300_model-4299.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model/model-/'))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
writer = tf.summary.FileWriter("TensorBoard/", graph = graph)
Input_sketch = graph.get_tensor_by_name("Placeholder_1:0")
op_to_restore = graph.get_tensor_by_name("generator/conv2d_transpose_6/Tanh:0")

#Now, access the op that you want to run.

result = sess.run(op_to_restore, feed_dict={Input_sketch: img})

plot(result)
