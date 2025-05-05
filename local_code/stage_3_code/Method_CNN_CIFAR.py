'''
Concrete MethodModule class for CNN training on CIFAR-10
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.method import method
from local_code.stage_3_code.Evaluate_Metrics import Evaluate_Metrics
from local_code.stage_3_code.Training_Visualization import Training_Visualizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class Method_CNN_CIFAR(method, nn.Module):
    data = None
    # it defines the max rounds to train the model
    max_epoch = 50
    # it defines the learning rate for gradient descent based optimizer for model learning
    learning_rate = 1e-3
    # it defines the batch size for training
    batch_size = 64

    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        # CNN architecture
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1) # 3->32
        self.activation_func_1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 32->64
        self.activation_func_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 64->128
        self.activation_func_3 = nn.ReLU()

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1) # 128->128       
        self.activation_func_4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 8 * 8, 512)  # 32x32 -> 16x16 -> 8x8
        self.activation_func_linear_1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.activation_func_linear_2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # First conv block
        c1 = self.activation_func_1(self.conv1(x))
        c2 = self.activation_func_2(self.conv2(c1))
        p1 = self.pool1(c2)
        
        # Second conv block
        c3 = self.activation_func_3(self.conv3(p1))
        c4 = self.activation_func_4(self.conv4(c3))
        p2 = self.pool2(c4)
        
        # Flatten and fully connected layers
        f = torch.flatten(p2, 1)  # flatten all dimensions except batch
        h1 = self.activation_func_linear_1(self.fc1(f))
        h1 = self.dropout(h1)
        h2 = self.activation_func_linear_2(self.fc2(h1))
        h2 = self.dropout(h2)
        y_pred = self.fc3(h2)
        
        return y_pred

    def train_model(self, X, y):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        loss_function = nn.CrossEntropyLoss()
        evaluator = Evaluate_Metrics('training evaluator', '')
        visualizer = Training_Visualizer()
        
        # Convert data to PyTorch tensors and normalize
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        
        # Print data shapes for debugging
        print(f'Input data shape before permute: {X.shape}')
        
        # transfer the data format from [N, H, W, C] to [N, C, H, W]
        X = X.permute(0, 3, 1, 2)
        print(f'Input data shape after permute: {X.shape}')
        
        # Check if the shape is correct
        if X.shape[1] != 3:
            raise ValueError(f'Expected 3 channels, got {X.shape[1]} channels')
        
        # Normalize the data
        mean = X.mean(dim=(0, 2, 3), keepdim=True)
        std = X.std(dim=(0, 2, 3), keepdim=True)
        X = (X - mean) / std
        
        # Create data loader with smaller batch size
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Test run before training
        print("\nPerforming test run...")
        try:
            # Get a single batch
            test_batch = next(iter(train_loader))
            test_images, test_labels = test_batch
            
            # Test forward pass
            print("Testing forward pass...")
            test_outputs = self.forward(test_images)
            print(f"Forward pass successful. Output shape: {test_outputs.shape}")
            
            # Test backward pass
            print("Testing backward pass...")
            test_loss = loss_function(test_outputs, test_labels)
            test_loss.backward()
            print("Backward pass successful.")
            
            # Test evaluation
            print("Testing evaluation...")
            self.eval()
            with torch.no_grad():
                eval_outputs = self.forward(test_images)
                eval_pred = eval_outputs.max(1)[1]
                print(f"Evaluation successful. Predictions shape: {eval_pred.shape}")
            self.train()
            
            print("Test run completed successfully!\n")
        except Exception as e:
            print(f"Test run failed with error: {str(e)}")
            raise e
        
        # Training loop
        for epoch in range(self.max_epoch):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in tqdm(train_loader):
                # Check batch data shape
                if images.shape[1] != 3:
                    raise ValueError(f'Batch data shape incorrect: {images.shape}')
                
                # Forward pass
                outputs = self.forward(images)
                loss = loss_function(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Calculate statistics
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            
            # Calculate epoch statistics
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100. * correct / total
            
            # Record loss and accuracy for visualization
            visualizer.loss_history.append(epoch_loss)
            visualizer.acc_history.append(epoch_acc)
            
            # Print progress
            if epoch % 5 == 0:
                print(f'Epoch: {epoch}')
                print(f'Training Loss: {epoch_loss:.4f}')
                print(f'Training Accuracy: {epoch_acc:.2f}%\n')
                
                # Evaluate metrics on a subset of data
                self.eval()  # Set to evaluation mode
                with torch.no_grad():
                    # Use a subset of data for evaluation to save memory
                    eval_size = min(1000, len(X))
                    eval_indices = torch.randperm(len(X))[:eval_size]
                    eval_X = X[eval_indices]
                    eval_y = y[eval_indices]
                    
                    # Get predictions
                    eval_outputs = self.forward(eval_X)
                    eval_pred = eval_outputs.max(1)[1]
                    
                    # Update evaluator
                    evaluator.data = {
                        'true_y': eval_y.cpu().numpy(),
                        'pred_y': eval_pred.cpu().numpy()
                    }
                    
                    # Get scores
                    scores = evaluator.evaluate()
                    print('Training Metrics:')
                    for metric_name, score in scores.items():
                        print(f'  {metric_name}: {score:.4f}')
                    print()
                
                self.train()  # Set back to training mode
        
        # Plot training curves
        visualizer.plot_training_curve()

    def test(self, X):
        # do the testing, and result the result
        with torch.no_grad():
            X = torch.FloatTensor(X)
            # transfer the data format from [N, H, W, C] to [N, C, H, W]
            X = X.permute(0, 3, 1, 2)
            y_pred = self.forward(X)
            # convert the probability distributions to the corresponding labels
            return y_pred.max(1)[1]
    
    def run(self):
        print('method running...')
        print('--start training...')
        self.train_model(self.data['train']['X'], self.data['train']['y'])
        print('--start testing...')
        self.eval()  # Set to evaluation mode
        pred_y = self.test(self.data['test']['X'])
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}