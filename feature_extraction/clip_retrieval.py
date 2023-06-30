import os
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import open_clip


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


filename = 'C:\\Users\\pavon\\Downloads\\Grocery_products\\Training\\Food\\Pasta\\16.jpg'


if __name__ == '__main__':

    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    model, _, preprocess = open_clip.create_model_and_transforms('RN101',
                                                                 pretrained='openai')
    model.eval()
    model.to(device)

    #Load filename image, process it using Dino, then find the 5 most similar feature vectors in the 'database' folder
    #Load the image
    img = Image.open(filename)

    #Apply the weights to the image
    img = preprocess(img)
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(img).float()

    # Flatten the features
    image_features = torch.flatten(image_features)

    #Find the 5 most similar feature vectors in the database
    #Initialize the list of distances
    distances = []
    #Iterate over the database
    config = configparser.ConfigParser()
    config.read('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\config.ini')
    retrieval_path = config['Paths']['retrieval_clip']
    training_path = config['Paths']['train_data_path']
    for im_name in os.listdir(retrieval_path):
        #Load the feature vector
        features2 = torch.load(retrieval_path + im_name)
        #Compute the l2 distance
        #dist = torch.dist(features, features2)

        # cosine distance
        dist = 1 - F.cosine_similarity(image_features, features2.unsqueeze(0))

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