# CLASSIFICATION PART USING RESNET AND FREIBURG GROCERIES DATASET

import multiprocessing
import os
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import ToTensor

import utils
from finetune import config

dataset_dir = config['Paths']['freigburg_path']  # dataset path
label_map = {
    'BEANS': 0,
    'CAKE': 1,
    'CANDY': 2,
    'CEREAL': 3,
    'CHIPS': 4,
    'CHOCOLATE': 5,
    'COFFEE': 6,
    'CORN': 7,
    'FISH': 8,
    'FLOUR': 9,
    'HONEY': 10,
    'JAM': 11,
    'JUICE': 12,
    'MILK': 13,
    'NUTS': 14,
    'OIL': 15,
    'PASTA': 16,
    'RICE': 17,
    'SODA': 18,
    'SPICES': 19,
    'SUGAR': 20,
    'TEA': 21,
    'TOMATO_SAUCE': 22,
    'VINEGAR': 23,
    'WATER': 24
}


class FreiburgDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir  # directory dataset
        self.images = []  # images
        self.labels = []  # relative labels
        self._load_data()  # loading dataset

    def _load_data(self):
        label_names = sorted(os.listdir(self.data_dir))
        for label in label_names:
            label_dir = os.path.join(self.data_dir, label)
            if os.path.isdir(label_dir):
                image_files = os.listdir(label_dir)
                for img in image_files:
                    image_path = os.path.join(label_dir, img)
                    pil_image = Image.open(image_path).convert("RGB")
                    transform = ToTensor()
                    tensor_image = transform(pil_image)
                    self.images.append(tensor_image)
                tmp_labels = [label] * len(image_files)
                labels = [label_map[label] for label in tmp_labels]
                self.labels.extend(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):  # get a specific image given the index
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


def get_classification_model(num_classes):
    classifier = resnet50(weights=ResNet50_Weights.DEFAULT)
    num_features = classifier.fc.in_features
    classifier.fc = nn.Linear(num_features, num_classes)
    return classifier


def train_one_epoch(data_loader, model, optimizer, criterion, epoch, num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(data_loader):
        images, labels = batch

        print(f"Batch [{batch_idx + 1}/{len(data_loader)}]")  # print batch_index

        # Convert tuples to tensors
        images = torch.stack(images).to(device)
        labels = torch.Tensor(labels).long().to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(data_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}]: Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), 'model_weights_resnet50.pth')


def evaluate_model(data_loader_test, model):
    predicted_labels = []
    true_labels = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader_test):
            images, labels = batch
            print(f"Batch [{batch_idx + 1}/{len(data_loader_test)}]")  # print batch_index

            images = torch.stack(images).to(device)
            labels = torch.Tensor(labels).long().to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predicted_labels, true_labels


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # CREATION OF THE DATASET
    dataset = FreiburgDataset(dataset_dir)

    # TRAIN TEST SPLIT

    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    test_split = 0.2
    tsize = int(len(dataset) * test_split)
    dataset_train = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset, indices[-tsize:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    print('The training set has length = ', len(dataset))
    print('The test set has length = ', len(dataset_test))

    # TRAINING PART
    print('\nTraining part')

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 25

    # get the model using our helper function
    model = get_classification_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss()

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # training for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        train_one_epoch(data_loader, model, optimizer, criterion, epoch, num_epochs)

    # TEST PART
    print('\nTest part')
    predicted_labels, true_labels = evaluate_model(data_loader_test, model)

    # Computing performance measures
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Plotting Confusion Matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
