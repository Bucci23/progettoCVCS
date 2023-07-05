import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import n_of_shelves_class as nsc
from little_nn import FeedForwardNet
X_dataset, y_dataset = nsc.get_x_and_y()
y_dataset = y_dataset-4
# print(y_dataset)
train_data = X_dataset[:53]
train_labels = y_dataset[:53]
test_data = X_dataset[53:]
test_labels = y_dataset[53:]

# Convert the data and labels to PyTorch tensors
train_data = torch.from_numpy(train_data).float()
train_labels = torch.from_numpy(train_labels).long()
test_data = torch.from_numpy(test_data).float()
test_labels = torch.from_numpy(test_labels).long()
model = FeedForwardNet()
model = torch.load('model_53.pth')
#Evaluate both with MAE and Accuracy on both test set and train set:
with torch.no_grad():
    outputs = model(test_data)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    from sklearn.metrics import mean_absolute_error
    print('MAE:',mean_absolute_error(test_labels, predicted))
    print('Test accuracy:', accuracy)