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
import sys
sys.path.append('C:\\Users\\david\\PycharmProjects\\progettoTriste\\')
import SingleObjectDetectionCanny
filename = 'data\\uniform\\IMG_20230413_184440.jpg'
if __name__ == '__main__':
    #Load filename image, process it using resnet50 and the default pretrained weights, then find the 5 most similar feature vectors in the 'database' folder
    #Load the image
    img = read_image(filename)
    test_set = load_dataset.load_json_annotations_testing('C:\\Users\\david\\PycharmProjects\\progettoTriste\\dataset_annotations.json')
    if os.path.basename(filename) in test_set.keys():
        bbox = test_set[os.path.basename(filename)]
        print('Image found. It has been cropped!')
    else:
        print('Image not found. Cropping using single object detection...')
        bbox = SingleObjectDetectionCanny.single_img_detection(filename)
    img = load_dataset.crop_image(img, bbox)
    #Load the weights
    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms(antialias=True)
    #Apply the weights to the image
    img_transformed = preprocess(img)
    #Load the model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    #Set the model to evaluation mode
    model.eval()
    #Load the feature extractor
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    #Extract the features from the image
    with torch.no_grad():
        features = feature_extractor(img_transformed.unsqueeze(0))
    #Flatten the features
    features = torch.flatten(features)
    #Find the 5 most similar feature vectors in the database
    #Initialize the list of distances
    distances = []
    #Iterate over the database
    for im_name in os.listdir('retrieval_database_cropped\\'):
        #Load the feature vector
        features2 = torch.load('retrieval_database_cropped\\' + im_name)
        #Compute the l2 distance
        dist = torch.dist(features, features2)
        #Add the distance to the list
        distances.append((im_name, dist))
    #Sort the list
    distances.sort(key=lambda x: x[1])
    #Print the 5 most similar images
    for i in range(5):
        print(distances[i])
    #Plot the image and the 5 most similar images

    '''fig, a = plt.subplots(1, 6)
    fig.set_size_inches(10, 10)
    fig.show()
    a.show()'''