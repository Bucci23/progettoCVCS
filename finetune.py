import configparser
import multiprocessing
import os
import json
import csv
import random
import numpy as np
#import pandas as pd
import cv2
import plot_data
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils as utils
import transforms as T

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import annotate_groceries

config = configparser.ConfigParser()
config.read('config.ini')

# get the path to training data
#train_data_path = config['Paths']['train_data_path']

training_dir = config['Paths']['train_data_path']
test_dir = config['Paths']['train_data_path']
file_path = config['Paths']['file_path']


def load_bbs(filename):
    with open(filename, 'r') as file:
        d = json.load(file)
    return d


def load_csv_annotations(dir, test_names):
    annotations = {}
    for store in os.listdir(dir):
        s = os.path.join(dir, store)
        if not os.path.isdir(s):
            continue
        s = os.path.join(s, 'annotation/')
        for filename in os.listdir(s):
            if filename.endswith('.csv'):
                image = os.path.splitext(filename)[0]
                filename = os.path.join(s, filename)
                with open(filename, newline='') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
                    image_annotation = []
                    next(reader)
                    for row in reader:
                        single_bb = []
                        for field in row:
                            single_bb.append(field)
                        image_annotation.append(single_bb)
                filename = filename.replace('annotation', 'images')
                filename = filename.replace('.csv', '.jpg')
                annotations[filename] = image_annotation
    return annotations


class GroceryDataset(Dataset):
    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        lines = []
        self.bbs = load_bbs('grocery_output.json')
        self.classes = {}
        annotation_path = config['Paths']['annotation_path']
        with open('robaccia/grocery/Grocery_products/Training/classes.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in reader:
                self.classes[row[1]] = row[0]
        test_path = config['Paths']['test_path']
        test_file_path = config['Paths']['test_file_path']
        annotated_dict = load_csv_annotations('robaccia/grocery/Grocery_products/Testing/',
                                              'robaccia/grocery/Grocery_products/TestFiles.txt')
        self.imgs = list(annotated_dict.keys())
        self.annotations = annotated_dict
        print(len(self.classes))

    def annotate(self):
        return annotate_groceries.single_object_detect(self.imgs, 'robaccia/grocery_BBs', 105, 236)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        print(img_name)
        img = cv2.imread(img_name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        boxes = []
        labels = []
        relative_boxes = []
        wt = img.shape[1]
        ht = img.shape[0]
        specific_annot = self.annotations[img_name]
        for obj in specific_annot:
            labels.append(int(obj[0]))
            xmin = float(obj[2]) * self.width
            xmax = float(obj[3]) * self.width
            ymin = float(obj[4]) * self.height
            ymax = float(obj[5]) * self.height
            if xmin<0:
                xmin = 0
                rel_xmin = 0
            if ymin<0:
                ymin = 0
                rel_ymin = 0
            if xmax < 0:
                xmax = 0
                rel_xmax = 0
            if ymax<0:
                ymax = 0
                rel_ymax = 0
            rel_xmin = float(obj[2])
            rel_xmax = float(obj[3])
            rel_ymin = float(obj[4])
            rel_ymax = float(obj[5])
            boxes.append([xmin, ymin, xmax, ymax])
            relative_boxes.append([rel_xmin, rel_ymin, rel_xmax, rel_ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            sample = self.transforms(image=img_res,
                                     bboxes=relative_boxes,
                                     labels=labels)

            img_res = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])

        return img_res, target

    def __len__(self):
        return len(self.imgs)


def get_object_detection_model(num_classes):
    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# Send train=True from training transforms and False for val/test transforms
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(0.5),
            # ToTensorV2 converts image to pytorch tensor without div by 255
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

if __name__ == '__main__':    
    multiprocessing.freeze_support()
    # loading the dataset
    dataset = GroceryDataset(training_dir, 512, 512, transforms=get_transform(train=True))
    for i in range(len(dataset)):
        _, target = dataset[i]
        print(target)
    dataset_test = GroceryDataset(training_dir, 512, 512, transforms=get_transform(train=False))
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    
    # train test split
    test_split = 0.2
    tsize = int(len(dataset) * test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-tsize:])
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=10, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=10, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    print('this training set has length = ', len(dataset))
    print('the test set has length = ',len(dataset_test))
    
    #TRAINING:
    
    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    num_classes = 81
    
    # get the model using our helper function
    model = get_object_detection_model(num_classes)
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    
    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # training for 10 epochs
    num_epochs = 10
    
    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
    