from little_nn import FeedForwardNet
import torch
import torch.nn as nn
import torch.optim as optim
import get_fv
import n_of_shelves_class as nsc
from sklearn.metrics import mean_absolute_error,accuracy_score
class Big_NN(nn.Module):
    def __init__(self, n_input=340,n_hidden=64, n_output=5):
        super(Big_NN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 256)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, n_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
class Bigger_NN(nn.Module):
    def __init__(self, n_input=340,n_hidden=64, n_output=5):
        super(Bigger_NN, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, n_output)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = self.fc6(x)
        return x
if __name__ == '__main__':
    X_dataset, y_dataset = get_fv.get_double_x_and_y()
    # print(y_dataset)
    X_test, y_test = nsc.get_double_x_and_y(int(X_dataset.shape[1]/2))
    print(X_dataset.shape, X_test.shape)
    train_data = X_dataset 
    train_labels = y_dataset - 1
    test_data = X_test
    test_labels = y_test - 1
    #Make the test set the same shape as the training set in dimension 1 by padding with zeros
    
    # Convert the data and labels to PyTorch tensors
    train_data = torch.from_numpy(train_data).float()
    train_labels = torch.from_numpy(train_labels).long()
    test_data = torch.from_numpy(test_data).float()
    test_labels = torch.from_numpy(test_labels).long()
    test_data = torch.nn.functional.pad(test_data, (0,train_data.shape[1]-test_data.shape[1]), 'constant', 0)
    model = Bigger_NN(1198,512,10)
    #model = FeedForwardNet(1198,128,10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    num_epochs = 30
    batch_size = 32
    mae = []
    acc = []
    models = []
    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            # Extract a batch of data
            batch_data = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]

            # Forward pass
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels.reshape(-1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        with torch.no_grad():
            outputs = model(test_data)
            _, predicted = torch.max(outputs.data, 1)
            #vaccuracy = (predicted == test_labels).sum().item() / len(test_labels)
            mae.append(mean_absolute_error(test_labels, predicted))
            acc.append(accuracy_score(test_labels, predicted))
            outputs = model(train_data)
            _, predicted = torch.max(outputs.data, 1)
            #vaccuracy = (predicted == test_labels).sum().item() / len(test_labels)
            print(mean_absolute_error(train_labels, predicted))
            print(accuracy_score(train_labels, predicted))
            
        models.append(model)
    #Save the model with best accuracy
    best_model = models[acc.index(max(acc))]
    torch.save(best_model.state_dict(), 'big_model.pth')
    #print the best accuracy and its index
    print('Best accuracy:', max(acc), 'epoch:', acc.index(max(acc))+1)
    #print the best mae
    print('Best mae:', min(mae), 'epoch:', mae.index(min(mae))+1)
    #Compute mae on the test_labels
