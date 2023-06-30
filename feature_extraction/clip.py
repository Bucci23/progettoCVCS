import torch
from PIL import Image
import open_clip
import load_dataset
import torchvision
import torchvision.transforms as transforms
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

    print(open_clip.list_pretrained())

    #model, _, preprocess = open_clip.create_model_and_transforms('RN101', pretrained='openai')

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

    model.eval()
    model.to(device)

    for im_name in dataset.keys():
        if not os.path.exists(im_name):
            print('Image not found. Skipping: ', im_name)
            continue
        img = Image.open(im_name)
        img = preprocess(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(img).float()

        #Flatten the features
        image_features = torch.flatten(image_features)

        im_name_without_path = os.path.basename(im_name)

        dump_path = os.path.join('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\retrieval_clip_vit',
                                 im_name[42:].replace('\\', '-'))

        torch.save(image_features, dump_path + '.pt')
        print('Saved: ', dump_path)