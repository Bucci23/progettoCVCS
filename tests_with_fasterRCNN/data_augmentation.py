# DATA AUGMENTATION PART

import configparser
import multiprocessing
import os
import uuid

from PIL import Image
import matplotlib.pyplot as plt
import json
import csv
import random
import numpy as np
# import pandas as pd
import cv2
import plot_data
from torch.utils.data import Dataset
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights, \
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, \
    fasterrcnn_mobilenet_v3_large_fpn, \
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, \
    fasterrcnn_mobilenet_v3_large_320_fpn

from engine import train_one_epoch, evaluate
import utils as utils
import transforms as T
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import annotate_groceries
from finetune import GroceryDataset, training_dir

'''''
def show_img(img):
    # Convert the tensor to a NumPy array
    numpy_image = img.permute(1, 2, 0).detach().cpu().numpy()

    # Display the image using matplotlib
    plt.imshow(numpy_image)
    plt.axis('off')
    plt.show()
'''''

cont = 0  # global counter for image name


# Send train=True from training transforms and False for val/test transforms
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(True, p=0.7),  # horizontal flip
            A.RandomRotate90(p=0.5),  # rotation of 90°
            A.ChannelShuffle(p=0.8),  # mixing RGB channels
            ToTensorV2(p=1.0)  # ToTensorV2 converts image to pytorch tensor without div by 255
        ], bbox_params={'format': 'albumentations', 'label_fields': ['labels']})
    else:  # test part (NOT useful)
        return A.Compose([
            # ToTensorV2 converts image to pytorch tensor without div by 255
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'albumentations', 'label_fields': ['labels']})


def save_img_bb_and_labels(img, dict_boxes_labels): # function for saving an image and its bounding boxes and labels
    global cont
    tensor_to_image = transforms.ToPILImage()
    pil_image = tensor_to_image(img)

    directory = "augmented_data"
    filename = 'Image' + str(cont)
    image_path = os.path.join(directory, "{}.jpg".format(filename))
    # Save the image
    pil_image.save(image_path)

    # Save the bounding boxes and labels in a json file
    bb = dict_boxes_labels['boxes'].tolist()
    labels = dict_boxes_labels['labels'].tolist()

    bb_labels_path = os.path.join(directory, "{}.json".format(filename))
    with open(bb_labels_path, 'w') as json_file:
        json.dump(bb, json_file)
        json.dump(labels, json_file)
    cont += 1


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # new directory for data augmentation
    augmented_dir = 'augmented_data'
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)

    # data_augmentation of the GroceryDataset
    augmented_dataset = GroceryDataset(training_dir, 512, 512, transforms=get_transform(train=True))

    # saving results
    for tuple_img in augmented_dataset:
        save_img_bb_and_labels(tuple_img[0], tuple_img[1])
