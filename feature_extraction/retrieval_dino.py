import os
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F


def plot_images(filename, paths):
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
        #print(distances[i][0])
        # Replace "-" with "\\" and remove the ".pt" extension
        new_path = train + distances[i][0].replace("-", "\\").replace(".pt", "")
        #print(new_path)
        if "Jars\\Cans" in new_path:
            new_path = new_path.replace("Jars\\Cans", "Jars-Cans")
        if "Oil\\Vinegar" in new_path:
            new_path = new_path.replace("Oil\\Vinegar", "Oil-Vinegar")
        paths.append(new_path)

    return paths


filename = 'C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\data\\uniform\\IMG_20230413_111136.jpg'


if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)


    #Load filename image, process it using Dino, then find the 5 most similar feature vectors in the 'database' folder
    #Load the image
    img = Image.open(filename)

    normalize = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    #Apply the weights to the image
    img_transformed = normalize(img)
    img_transformed = img_transformed.unsqueeze(0).to(device)


    # model weights paths
    weights_path = 'Dino/dino_resnet50_pretrain.pth'

    weights = torch.load(weights_path, map_location=device)
    # load pretrained model for Dino --> Resnet50
    model = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    model.load_state_dict(weights)
    model.eval()


    #Set the model to evaluation mode
    model.eval()
    model.to(device)

    with torch.no_grad():
        features = model(img_transformed)

    # Flatten the features
    features = torch.flatten(features)

    #Find the 5 most similar feature vectors in the database
    #Initialize the list of distances
    distances = []
    #Iterate over the database
    config = configparser.ConfigParser()
    config.read('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\config.ini')
    retrieval_path = config['Paths']['retrieval_dino']
    training_path = config['Paths']['train_data_path']
    for im_name in os.listdir(retrieval_path):
        #Load the feature vector
        features2 = torch.load(retrieval_path + im_name)
        #Compute the l2 distance
        #dist = torch.dist(features, features2)

        # cosine distance
        dist = 1 - F.cosine_similarity(features, features2.unsqueeze(0))

        #Add the distance to the list
        distances.append((im_name, dist))
    #Sort the list
    distances.sort(key=lambda x: x[1])
    #Print the 5 most similar images
    for i in range(10):
        print(distances[i])

    #Plot the image and the 5 most similar images

    paths = []
    paths = obtain_path(distances, training_path)
    #plot_images(filename, paths)
    plot_images_big(filename, paths)