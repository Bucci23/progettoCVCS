import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torchvision.transforms import transforms
import get_groceries
# Set random seed for reproducibility
torch.manual_seed(42)

# Define the transformation applied to the images
base_dir = '/work/cvcs_2023_group12/Grocery_products/Training'
# Create the custom dataset object
dataset = get_groceries.GroceryDataset(base_dir, 512, 512, get_groceries.get_transform(train=True)) # Replace with your custom dataset object

# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create data loaders for train and test sets
train_data_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
test_data_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

# Load the pre-trained ResNet-50 model
model = torch.load('resnet50aug_grocery_epoch14.pth')
# weights = torchvision.models.ResNet18_Weights.DEFAULT 
# model = torchvision.models.resnet18(weights=weights)
# Modify the classifier
num_classes = dataset.num_classes  # Replace with the number of classes in your custom dataset
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model = model.to(device)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = Adam(model.parameters(), lr=0.0005)

# Training loop
num_epochs = 100
print("Starting training loop...")
for epoch in range(14,num_epochs, 1):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    model.train()
    for images, labels in train_data_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_data_loader)
    train_accuracy = correct_predictions / total_predictions

    # Evaluation on test set
    model.eval()
    test_correct_predictions = 0
    test_total_predictions = 0
    with torch.no_grad():
        for test_images, test_labels in test_data_loader:
            test_images = test_images.to(device)
            test_labels = test_labels.to(device)

            test_outputs = model(test_images)
            _, test_predicted = torch.max(test_outputs.data, 1)
            test_total_predictions += test_labels.size(0)
            test_correct_predictions += (test_predicted == test_labels).sum().item()

    test_accuracy = test_correct_predictions / test_total_predictions
    file = open("resnet50aug_groceryLOGS.txt", "a")
    file.write(f"Epoch [{epoch+1}/{num_epochs}]\n")
    file.write(f"Train Loss: {epoch_loss:.4f}\n")
    file.write(f"Train Accuracy: {train_accuracy:.4f}\n")
    file.write(f"Test Accuracy: {test_accuracy:.4f}\n")
    file.write("-" * 50)
    file.close()
    #save the model:
    torch.save(model, f'resnet50aug_grocery_epoch{epoch}.pth')
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {epoch_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("-" * 50)

