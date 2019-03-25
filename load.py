import numpy as np
import os
import cv2 as cv

def load_batch(self, batch_size=1):
    data = glob("./new_train_test_data/train_edge_data/*")
    label = glob("./new_train_test_data/train_edge_label/*")
    self.n_batches = int(len(data) / batch_size)
    for i in range(self.n_batches-1):
        edge = []
        input_sketch = []
        for filename in data[i:i+batch_size]:
            img = cv.imread(filename, cv.IMREAD_COLOR)/255
            img = cv.resize(img, (128, 128))
            input_sketch.append(img)
        for filename in label[i:i+batch_size]:
            img = cv.imread(filename, cv.IMREAD_COLOR)/255
            img = cv.resize(img, (128, 128))
            edge.append(img)
        edge = np.array(edge)
        input_sketch = np.array(input_sketch)
        yield edge, input_sketch
def load_data(self, batch_size=1):
    data = glob("./new_train_test_data/test_edge_data/*")
    label = glob("./new_train_test_data/test_edge_label/*")
    batch_data = np.random.choice(data, size=batch_size)
    batch_label = np.random.choice(label, size=batch_size)

    edge = []
    input_sketch = []
    for filename in batch_data:
        img = cv.imread(filename, cv.IMREAD_COLOR)/255
        img = cv.resize(img, (128, 128))
        input_sketch.append(img)
    for filename in batch_label:
        img = cv.imread(filename, cv.IMREAD_COLOR)/255
        img = cv.resize(img, (128, 128))
        edge.append(img)

    edge = np.array(edge)
    input_sketch = np.array(input_sketch)

    return edge, input_sketch
