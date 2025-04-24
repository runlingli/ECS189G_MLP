'''
Concrete Evaluate class for multiple evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class Evaluate_Metrics(evaluate):
    data = None
    
    def evaluate(self):
        print('evaluating performance...')
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        
        accuracy = accuracy_score(true_y, pred_y)
        f1_weighted = f1_score(true_y, pred_y, average='weighted')
        f1_macro = f1_score(true_y, pred_y, average='macro')
        f1_micro = f1_score(true_y, pred_y, average='micro')
        
        recall_weighted = recall_score(true_y, pred_y, average='weighted')
        recall_macro = recall_score(true_y, pred_y, average='macro')
        recall_micro = recall_score(true_y, pred_y, average='micro')
        
        precision_weighted = precision_score(true_y, pred_y, average='weighted')
        precision_macro = precision_score(true_y, pred_y, average='macro')
        precision_micro = precision_score(true_y, pred_y, average='micro')
        
        return {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'f1_micro': f1_micro,
            'recall_weighted': recall_weighted,
            'recall_macro': recall_macro,
            'recall_micro': recall_micro,
            'precision_weighted': precision_weighted,
            'precision_macro': precision_macro,
            'precision_micro': precision_micro
        } 