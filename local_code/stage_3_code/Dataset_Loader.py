'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np
from matplotlib import pyplot as plt


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    dataset_name = None  # additional parameter: specify the dataset name
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        self.dataset_name = dName  # name of the dataset
    
    def load(self):
        print(f'loading {self.dataset_name} dataset...')
        
        # load the dataset based on the dataset name
        if self.dataset_name == 'ORL':
            return self._load_orl()
        elif self.dataset_name == 'CIFAR':
            return self._load_cifar()
        elif self.dataset_name == 'MNIST':
            return self._load_mnist()
        else:
            raise ValueError(f'Unknown dataset: {self.dataset_name}')
    
    def _load_orl(self):
        """load the ORL dataset"""
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        
        # load the training data
        train_images = np.array([instance['image'] for instance in data['train']])
        train_labels = np.array([instance['label'] for instance in data['train']])
        
        # load the test data
        test_images = np.array([instance['image'] for instance in data['test']])
        test_labels = np.array([instance['label'] for instance in data['test']])
        
        return {'X': train_images, 'y': train_labels, 'test_X': test_images, 'test_y': test_labels}
    
    def _load_cifar(self):
        """load the CIFAR dataset"""
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        
        # load the training data
        train_images = np.array([instance['image'] for instance in data['train']])
        train_labels = np.array([instance['label'] for instance in data['train']])
        
        # load the test data
        test_images = np.array([instance['image'] for instance in data['test']])
        test_labels = np.array([instance['label'] for instance in data['test']])
        
        return {'X': train_images, 'y': train_labels, 'test_X': test_images, 'test_y': test_labels}
    
    def _load_mnist(self):
        """load the MNIST dataset"""
        f = open(self.dataset_source_folder_path + self.dataset_source_file_name, 'rb')
        data = pickle.load(f)
        f.close()
        
        # load the training data
        train_images = np.array([instance['image'] for instance in data['train']])
        train_labels = np.array([instance['label'] for instance in data['train']])
        
        # load the test data
        test_images = np.array([instance['image'] for instance in data['test']])
        test_labels = np.array([instance['label'] for instance in data['test']])
        
        return {'X': train_images, 'y': train_labels, 'test_X': test_images, 'test_y': test_labels}
    