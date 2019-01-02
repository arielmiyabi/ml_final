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

def plot(samples):
    plt.cla()
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        f_sample = np.asarray(sample).flatten()
        normalized = (f_sample-min(f_sample))/(max(f_sample)-min(f_sample))
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(normalized.reshape(imgsize, imgsize), cmap='Greys_r')

    return fig
os.makedirs('result/test', exist_ok=True)
img = cv.imread(args.sketch, cv.IMREAD_GRAYSCALE)/255
img = cv.resize(img, (imgsize, imgsize))
img = np.reshape(img, [1, imgsize, imgsize, 1])

sess=tf.Session()
#First let's load meta graph and restore weights
saver = tf.train.import_meta_graph('./model/model-2/40_model-40.meta')
saver.restore(sess,tf.train.latest_checkpoint('./model/model-2/'))

# Now, let's access and create placeholders variables and
# create feed-dict to feed new data

graph = tf.get_default_graph()
writer = tf.summary.FileWriter("TensorBoard/", graph = graph)
Input_sketch = graph.get_tensor_by_name("Placeholder_1:0")
op_to_restore = graph.get_tensor_by_name("generator/conv2d_transpose_6/Tanh:0")

#Now, access the op that you want to run.

result = sess.run(op_to_restore, feed_dict={Input_sketch: img})

fig = plot(result)
plt.savefig('result/test/{}.png'.format(str(args.sketch.split('/')[-1]).zfill(3)), bbox_inches='tight')
plt.close(fig)

#This will print 60 which is calculated
#using new values of w1 and w2 and saved value of b1.
