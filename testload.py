import tensorflow as tf
import numpy as np
import cv2
from glob import glob


def load_batch(batch_size=1, img_res=(128, 128)):
    train_path = glob('./data/train_data/*')
    label_path = glob('./data/train_label/*')

    n_batches = int(len(train_path) / batch_size)

    for i in range(n_batches-1):
        train_batch = train_path[i*batch_size:(i+1)*batch_size]
        label_batch = label_path[i*batch_size:(i+1)*batch_size]
        imgs_A, imgs_B = [], []
        for train_img in train_batch:
            img_B = cv2.imread(train_img, cv2.IMREAD_COLOR)
            img_B = cv2.resize(img_B, img_res)
            imgs_B.append(img_B)

        for label_img in label_batch:
            img_A = cv2.imread(label_img, cv2.IMREAD_COLOR)
            img_A = cv2.resize(img_A, img_res)
            imgs_A.append(img_A)

        imgs_A = np.array(imgs_A).astype(np.float32) /255.0
        # print(imgs_A.shape)
        return np.array(imgs_A).astype(np.float32) /255.0, np.array(imgs_B).astype(np.float32) / 255.0


Edge = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
Input_sketch = tf.placeholder(tf.float32, shape=[None, 128, 128, 3])
out = tf.multiply(Edge, Input_sketch)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

edge, input_sketch = load_batch()
print (edge.dtype)
print (edge.shape)
print (input_sketch.dtype)
print (input_sketch.shape)

result = sess.run(out, feed_dict={Edge: edge, Input_sketch: input_sketch})
result = sess.run(out, feed_dict={Input_sketch: input_sketch})
