'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)
        self.dataset = None
        self.method = None
        self.result = None
        self.evaluate = None

    def prepare(self, dataset, method, result, evaluate):
        self.dataset = dataset
        self.method = method
        self.result = result
        self.evaluate = evaluate

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        
        X_train = loaded_data['X']  # 50000 sample
        y_train = loaded_data['y']
        X_test = loaded_data['test_X']  # 10000 
        y_test = loaded_data['test_y']

        print(f'training set size: {X_train.shape}')
        print(f'testing set size: {X_test.shape}')

        # set the training and testing data
        self.method.data = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }

        # run the training and testing
        learned_result = self.method.run()

        # save the result
        self.result.data = learned_result
        self.result.save()

        # evaluate the result
        self.evaluate.data = learned_result
        scores = self.evaluate.evaluate()

        return scores

        