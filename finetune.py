import os
import json
import csv
import random
import numpy as np
import pandas as pd
import cv2
import plot_data

import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# from myVision.engine import train_one_epoch, evaluate
import myVision.utils as utils
import myVision.transforms as T

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import annotate_groceries

training_dir = 'robaccia/grocery/grocery_products/'
test_dir = 'robaccia/grocery/grocery_products/'
file_path = 'robaccia/grocery/grocery_products/TrainingFiles.txt'


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


class GroceryDataset(torch.utils.data.Dataset):
    def __init__(self, files_dir, width, height, transforms=None):
        self.transforms = transforms
        self.files_dir = files_dir
        self.height = height
        self.width = width
        lines = []
        self.bbs = load_bbs('grocery_output.json')
        self.classes = {}
        with open('robaccia/grocery/Grocery_products/Training/classes.csv', newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            next(reader)
            for row in reader:
                self.classes[row[1]] = row[0]
        annotated_dict = load_csv_annotations('robaccia/grocery/Grocery_products/Testing/',
                                              'robaccia/grocery/Grocery_products/TestFiles.txt')
        self.imgs = list(annotated_dict.keys())
        self.annotations = annotated_dict

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
        wt = img.shape[1]
        ht = img.shape[0]
        specific_annot = self.annotations[img_name]
        for obj in specific_annot:
            labels.append(int(obj[0]))
            xmin = float(obj[2]) * self.width
            xmax = float(obj[3]) * self.width
            ymin = float(obj[4]) * self.height
            ymax = float(obj[5]) * self.height
            boxes.append([xmin, ymin, xmax, ymax])

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
                                     bboxes=target['boxes'],
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


# loading the dataset
dataset = GroceryDataset(training_dir, 512, 512, transforms=get_transform(train=True))
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
# print('this dataset has length = ', len(dataset))

