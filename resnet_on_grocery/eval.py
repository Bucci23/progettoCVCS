import torch
import get_groceries
from torch.utils.data import DataLoader
import sklearn
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
# for every model in models, load the weights and evaluate on the test set computing accuracy and mean precision recall and f1 score for each of the classes. save the results in a csv file
for cur_model in models:
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
        model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        true_positives = [0]*num_classes
        false_positives = [0]*num_classes
        false_negatives = [0]*num_classes
        for images, labels in test_data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            for i in range(len(labels)):
                if predicted[i] == labels[i]:
                    true_positives[labels[i]] += 1
                else:
                    false_positives[predicted[i]] += 1
                    false_negatives[labels[i]] += 1
        accuracy = 100 * correct_predictions / total_predictions
        print('Test Accuracy: {:.2f}%'.format(accuracy))
        # compute precision, recall and f1 score
        precisions = [0]*num_classes
        recalls = [0]*num_classes
        f1_scores = [0]*num_classes
        for i in range(num_classes):
            precisions[i] = true_positives[i]/(true_positives[i]+false_positives[i]) if true_positives[i]+false_positives[i] > 0 else 0
            recalls[i] = true_positives[i]/(true_positives[i]+false_negatives[i]) if true_positives[i]+false_negatives[i] > 0 else 0
            f1_scores[i] = 2*(precisions[i]*recalls[i])/(precisions[i]+recalls[i]) if precisions[i]+recalls[i] > 0 else 0
        print('Precision for each class: ', precisions)
        print('Recall for each class: ', recalls)
        print('F1 score for each class: ', f1_scores)
        print('Mean precision: ', sum(precisions)/len(precisions))
        print('Mean recall: ', sum(recalls)/len(recalls))
        print('Mean f1 score: ', sum(f1_scores)/len(f1_scores))
        #save everything in a csv file
        with open('results.csv', 'a') as f:
            f.write(f'{cur_model},{accuracy},{sum(precisions)/len(precisions)},{sum(recalls)/len(recalls)},{sum(f1_scores)/len(f1_scores)}\n')

