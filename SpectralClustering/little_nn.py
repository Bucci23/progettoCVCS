import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import n_of_shelves_class as nsc

class FeedForwardNet(nn.Module):
    def __init__(self):
        super(FeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(340, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 5)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
if __name__ == '__main__':
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 10
    batch_size = 16

    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            # Extract a batch of data
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    with torch.no_grad():
        outputs = model(test_data)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == test_labels).sum().item() / len(test_labels)
    torch.save(model, 'model.pth')
    #Compute mae on the test_labels
    from sklearn.metrics import mean_absolute_error
    print('MAE:',mean_absolute_error(test_labels, predicted))
    print('Test accuracy:', accuracy)