'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting
import numpy as np

class Setting_Train_Test_Split(setting):
    fold = 3
    def __init__(self, sName=None, sDescription=None, dataset_test=None):
        super().__init__(sName, sDescription) 
        self.dataset_test = dataset_test  # change constructor

    def load_run_save_evaluate(self):
        
        loaded_train = self.dataset.load()
        X_train = loaded_train['X']
        y_train = loaded_train['y']

        loaded_test = self.dataset_test.load()
        X_test = loaded_test['X']
        y_test = loaded_test['y']

        self.method.data = {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }

        learned_result = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        scores = self.evaluate.evaluate()

        return scores

        