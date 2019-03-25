import numpy as np
from glob import glob
import cv2
import random
import scipy
import os

class Data_processor():
    def __init__(self):
        self.traindata_filenames = glob('distanceFiled_mask3_flip/*')
        self.labeldata_filenames = glob('distanceFiled_mask3_photo/*')
        self.photodata_filenames = glob('ShoeV2_photo/*')
        self.edge_filenames = glob('ShoeV2_photo_edge/*')
        self.label_dic = {}
        self.photo_dic = {}
        self.edge_dic = {}
        for labeldata_filename in self.labeldata_filenames:
            filename = str(str(labeldata_filename).split('/')[1]).split('.')[0]
            self.label_dic[filename] = labeldata_filename
        for photodata_filename in self.photodata_filenames:
            filename = str(str(photodata_filename).split('/')[1]).split('.')[0]
            self.photo_dic[filename] = photodata_filename
        for edge_filename in self.edge_filenames:
            filename = str(str(edge_filename).split('/')[1]).split('.')[0]
            self.edge_dic[filename] = edge_filename

    def rename_data(self):
        os.system('mkdir dataset')
        os.system('mkdir labelset')
        os.system('mkdir photoset')
        os.system('mkdir edgeset')
        random.shuffle(self.traindata_filenames)
        for i in range(0, len(self.traindata_filenames)):
            key  = str(str(str(self.traindata_filenames[i]).split('/')[1]).split('.')[0]).split('_')[0]
            labeldata_filename = self.label_dic[key]
            photo_filename = self.photo_dic[key]
            edge_filename = self.edge_dic[key]
            os.system('cp ' + self.traindata_filenames[i] + ' ./dataset/' + str(i) + '.png')
            os.system('cp ' + labeldata_filename + ' ./labelset/' + str(i) + '.png')
            os.system('cp ' + photo_filename + ' ./photoset/' + str(i) + '.png')
            os.system('cp ' + edge_filename + ' ./edgeset/' + str(i) + '.png')
    
    def split_train_and_test(self):
        dataset_filenames = glob('dataset/*')
        labelset_filenames = glob('labelset/*')

        os.system('mkdir train_data')
        os.system('mkdir train_label')
        os.system('mkdir test_data')
        os.system('mkdir test_label')
        for i in range(0, 6000):
            filename = str(dataset_filenames[i]).split('/')[1]
            os.system('cp ' + dataset_filenames[i] + ' ./train_data/' + filename)
            os.system('cp ' + labelset_filenames[i] + ' ./train_label/' + filename)
        for i in range(6000, len(dataset_filenames)):
            filename = str(dataset_filenames[i]).split('/')[1]
            os.system('cp ' + dataset_filenames[i] + ' ./test_data/' + filename)
            os.system('cp ' + labelset_filenames[i] + ' ./test_label/' + filename)



if __name__ == '__main__':
    dl = Data_processor()
    dl.rename_data()
    dl.split_train_and_test()


