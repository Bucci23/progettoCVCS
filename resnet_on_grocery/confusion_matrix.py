import torch
import get_groceries
from torch.utils.data import DataLoader
import sklearn
import matplotlib.pyplot as plt
import numpy as np
models = ['resnet18aug_grocery_epoch28.pth', 'resnet50_grocery_epoch10.pth','resnet50aug_grocery_epoch24.pth']
torch.manual_seed(42)

# Define the transformation applied to the images
base_dir = '/work/cvcs_2023_group12/Grocery_products/Training'
# Create the custom dataset object
dataset = get_groceries.GroceryDataset(base_dir, 512, 512, get_groceries.get_transform(train=False)) 
# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# Create data loaders for train and test sets
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
num_classes = dataset.num_classes

for cur_model in models:
    y_pred=np.array([])
    y_true =np.array([])
    #For each model in models, evaluate it on the test set, accumulate in an array all the predictions and in another array all the ground truth labels
    # Load the pre-trained ResNet-50 model
    model = torch.load(cur_model)
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = model.to(device)
    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()
    print("Starting testing loop...")
    with torch.no_grad():
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred = np.append(y_pred,predicted.cpu().numpy())
            y_true = np.append(y_true,labels.cpu().numpy())
    # Compute the confusion matrix using sklearn
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    # Normalize the confusion matrix
    print(cm, '\n')
    #save the confusion matrix in a csv file as integer values
    np.savetxt("confusion_matrix_"+cur_model+".csv", cm.astype(int), delimiter=",")
    #plot the confusion matrix
    # plt.figure(figsize=(10,10))
    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
