import torch
import torch.nn as nn
import load_dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import os
from torchvision.models import resnet50
import load_dataset


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    # load dataset
    dataset = load_dataset.load_img('C:\\Users\\pavon\\Downloads\\Grocery_products\\')

    # model weights paths
    weights_path = 'resnet50aug_grocery_epoch24.pth'

    #weights = torch.load(weights_path, map_location=device)
    #resnet50 = resnet50(weights=weights)
    #preprocess = resnet50.transforms()

    resnet50 = torch.load(weights_path)

    resnet50.eval()
    resnet50.to(device)
    print('Model loaded successfully')

    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])

    for im_name in dataset.keys():
        if not os.path.exists(im_name):
            print('Image not found. Skipping: ', im_name)
            continue
        img = Image.open(im_name)
        img = preprocess(img)
        img = img.unsqueeze(0).to(device)


        with torch.no_grad():
            features = feature_extractor(img)

        #Flatten the features
        features = torch.flatten(features)

        im_name_without_path = os.path.basename(im_name)

        dump_path = os.path.join('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\retrieval_finetune_resnet',
                                 im_name[42:].replace('\\', '-'))

        torch.save(features, dump_path + '.pt')
        print('Saved: ', dump_path)