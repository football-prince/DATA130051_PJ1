'''
Author: Zihao Cheng
Student number: 21307130080
School of data science

Homepage: https://github.com/football-prince/DATA130051_PJ1

This pyhton fuctuin implements a Fashion-Mnist data loader.
'''

import os
import struct
import numpy as np
import matplotlib.pyplot as plt

class MNISTDatasetLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
    
    def load_images(self, file_name):
        '''Load image files and reshape each image into a one-dimensional vector.'''
        file_path = os.path.join(self.dataset_path, file_name)
        with open(file_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            # Ensure each image is flattened into a long vector
            images = np.fromfile(file, dtype=np.uint8).reshape(size, rows * cols)
        return images

    def load_labels(self, file_name):
        '''Load label files.'''
        file_path = os.path.join(self.dataset_path, file_name)
        with open(file_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            labels = np.fromfile(file, dtype=np.uint8)
        return labels

    def load_dataset(self):
        '''Load training and testing dataset data.'''
        self.train_images = self.load_images('train-images-idx3-ubyte')
        self.train_labels = self.load_labels('train-labels-idx1-ubyte')
        self.test_images = self.load_images('t10k-images-idx3-ubyte')
        self.test_labels = self.load_labels('t10k-labels-idx1-ubyte')
        # normalization
        self.train_images = self.train_images.astype('float32') / 255.0
        self.test_images = self.test_images.astype('float32') / 255.0

if __name__ == '__main__':
    pass