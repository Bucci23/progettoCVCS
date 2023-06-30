import torch
import load_dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from PIL import Image
import os


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
    weights_path = 'Dino/dino_resnet50_pretrain.pth'
    #weights_path = 'Dino/dino_resnet50_pretrain_full_checkpoint.pth'
    #weights_path = 'Dino/dino_resnet50_linearweights.pth'

    weights = torch.load(weights_path, map_location=device)
    # load pretrained model for Dino --> Resnet50
    resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
    resnet50.load_state_dict(weights)
    resnet50.eval()
    resnet50.to(device)
    print('Model loaded successfully')

    normalize = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    """
    # Load and preprocess the image
    image_path = 'C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\data\\uniform\\IMG_20230413_110732.jpg'
    image = Image.open(image_path)
    input_data = normalize(image).unsqueeze(0)

    # Perform inference
    output = resnet50(input_data)
    print(output.shape)

    """

    for im_name in dataset.keys():
        if not os.path.exists(im_name):
            print('Image not found. Skipping: ', im_name)
            continue
        img = Image.open(im_name)
        img = normalize(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            features = resnet50(img)

        #Flatten the features
        features = torch.flatten(features)

        im_name_without_path = os.path.basename(im_name)

        dump_path = os.path.join('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\retrieval_dino',
                                 im_name[42:].replace('\\', '-'))

        torch.save(features, dump_path + '.pt')
        print('Saved: ', dump_path)