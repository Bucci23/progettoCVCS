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
import configparser
import copy
import load_dataset
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
import cv2
import sys
sys.path.append('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\')
import SingleObjectDetectionCanny
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_images(filename, paths, image_cropped):
    fig, axs = plt.subplots(2, 5)
    image = mpimg.imread(filename)
    images = [mpimg.imread(paths[i]) for i in range(5)]
    # Display the image
    axs[0, 0].imshow(image)
    axs[0, 0].set_aspect('auto')

    for i in range(5):
        axs[1, i].imshow(images[i])
        axs[1, i].axis('off')
        axs[0, i].axis('off')

    # Normalize the image tensor to the valid range
    image_cropped = torch.clamp(image_cropped, 0, 1)
    axs[0, 4].imshow(np.transpose(image_cropped, (1, 2, 0)))

    # Show the plot
    plt.show()


def plot_images_big(filename, paths):
    fig, axs = plt.subplots(2, 10)
    image = mpimg.imread(filename)
    images = [mpimg.imread(paths[i]) for i in range(10)]
    # Display the image
    axs[0, 0].imshow(image)
    axs[0, 0].set_aspect('auto')

    for i in range(10):
        axs[1, i].imshow(images[i])
        axs[1, i].axis('off')
        axs[0, i].axis('off')
    # Show the plot
    plt.show()


def obtain_path(distances, train):
    paths = []
    for i in range(10):
        print(distances[i][0])
        # Replace "-" with "\\" and remove the ".pt" extension
        new_path = train + distances[i][0].replace("-", "\\").replace(".pt", "")
        print(new_path)
        if "Jars\\Cans" in new_path:
            new_path = new_path.replace("Jars\\Cans", "Jars-Cans")
        if "Oil\\Vinegar" in new_path:
            new_path = new_path.replace("Oil\\Vinegar", "Oil-Vinegar")
        paths.append(new_path)

    return paths


filename = 'C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\data\\uniform\\IMG_20230413_184440.jpg'


if __name__ == '__main__':
    #Load filename image, process it using resnet50 and the default pretrained weights, then find the 5 most similar feature vectors in the 'database' folder
    #Load the image
    img = read_image(filename)
    test_set = load_dataset.load_json_annotations_testing('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\dataset_annotations.json')
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

    # da fare su database better

    for im_name in os.listdir('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\retrieval_database_better\\'):
        #Load the feature vector
        features2 = torch.load('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\retrieval_database_better\\' + im_name)
        #Compute the l2 distance
        dist = torch.dist(features, features2)

        # cosine distance
        # dist = 1 - F.cosine_similarity(features, features2.unsqueeze(0))

        #Add the distance to the list
        distances.append((im_name, dist))
    #Sort the list
    distances.sort(key=lambda x: x[1])
    #Print the 5 most similar images
    for i in range(10):
        print(distances[i])

    config = configparser.ConfigParser()
    config.read('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\config.ini')
    training_path = config['Paths']['train_data_path']
    #Plot the image and the 5 most similar images

    paths = []
    paths = obtain_path(distances, training_path)
    #plot_images(filename, paths, img_transformed)
    plot_images_big(filename, paths)