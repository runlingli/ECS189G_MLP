'''
Visualization module for MLP training process
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import os

class MLP_Training_Visualizer:
    def __init__(self, save_dir='../../result/stage_2_result/'):
        self.loss_history = []
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def train_with_visualization(self, model, X, y):
        optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        
        for epoch in range(model.max_epoch):
            y_pred = model.forward(torch.FloatTensor(np.array(X)))
            y_true = torch.LongTensor(np.array(y))
            train_loss = loss_function(y_pred, y_true)
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # record the loss value of each epoch
            self.loss_history.append(train_loss.item())
            
            if epoch % 50 == 0:
                print(f'Epoch: {epoch}, Loss: {train_loss.item():.4f}')
    
    def plot_training_curve(self, save_path='MLP_training_curve.png'):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_history, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('MLP Training Loss Curve')
        plt.legend()
        plt.grid(True)
        full_path = os.path.join(self.save_dir, save_path)
        plt.savefig(full_path)
        plt.close()
        
    def run_visualization(self, model, X, y):
        # train and record loss
        self.train_with_visualization(model, X, y)
        
        # generate and save training curve
        self.plot_training_curve()
        
        return model 