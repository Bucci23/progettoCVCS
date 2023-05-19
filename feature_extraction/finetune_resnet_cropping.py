from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights
import time
import os
import copy
import load_dataset
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import cv2

if __name__ == '__main__':
    dataset = load_dataset.load_img('robaccia\\grocery\\Grocery_products\\')
    # Initialize the Weight Transforms
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms(antialias=True)
    #Apply the weights to the image
    #Load the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    #Set the model to evaluation mode
    model.eval()
    annots = load_dataset.load_json_annotations('C:\\Users\\david\\PycharmProjects\\progettoTriste\\grocery_output.json')
    # Apply it to the input image
    for im_name in annots.keys():
        if not os.path.exists(im_name):
            print('Image not found. Skipping: ', im_name)
            continue
        if not im_name in annots.keys():
            print('Annotation not found. Skipping: ', im_name)
            continue
        img = read_image(im_name)
        bbox = annots[im_name]
        img = load_dataset.crop_image(img, bbox)
        img_transformed = preprocess(img)
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        #Extract the features from the image
        with torch.no_grad():
            features = feature_extractor(img_transformed.unsqueeze(0))
            #Flatten the features
        features = torch.flatten(features)
        #dump the transformed image to disk
        dump_path = os.path.join('retrieval_database_cropped\\', im_name.replace('\\', '-').replace('/', '-'))
        torch.save(features, dump_path + '.pt')
        print('Saved: ', dump_path) 
