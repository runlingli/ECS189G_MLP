'''
Concrete IO class for a specific dataset
'''

import pandas as pd
# use pandas to load the dataset
from local_code.base_class.dataset import dataset


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
    
    def load(self):
        print('loading data...')
        X = []
        y = []
        f = pd.read_csv(self.dataset_source_folder_path + self.dataset_source_file_name)
        # the first column is the label
        y = f.iloc[:, 0].values 
        # the rest of the columns are the features
        X = f.iloc[:, 1:].values 

        return {'X': X, 'y': y}