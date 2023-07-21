# From Object detection to Image Captioning in Retail Environment
This repository contains the reference code for the computer vision and cognitive systems project exam:

We will study the problems of object detection, image classification and content based retrieval at a supermarket using both classical computer vision and deep neural network techniques. In our paper we present a complete pipeline for all of these tasks, in addition we identify object positions in the scene and generate a  natural language description of it. For the object detection pipeline we used a regional convolutional neural network to localize objects in the image, then a feed forward network to predict the number of shelves present in the picture. Finally we performed spatial spectral clustering to identify the products in their specific shelf. Regarding the classification and retrieval part we used a contrastive language-image pretraining (CLIP) neural network to generate embeddings and implement zero-shot classification, along with a classical convolutive classification approach, to then compare the various results. Lastly we produced captions for our images using CoCa (Contrastive Captioner).


# Structure
Feature extraction --> retrieval pipeline
<br>
Classification-Captioning --> classification/captioning pipeline
Resnet_on_grocery --> classification on Grocery dataset
<br>
Spectral Clustering --> object shelf detection/clustering
<br>
FasterRCNN_on_SKU110K --> object detection, finetuning
<br>
Models --> file.pth dei vari modelli


