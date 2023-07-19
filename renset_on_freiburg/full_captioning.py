import os
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mpmath
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
import open_clip
from open_clip import tokenizer
import glob
import itertools
from sklearn import metrics
from sklearn.metrics import ConfusionMatrixDisplay
from torchmetrics.multimodal import CLIPScore


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    config = configparser.ConfigParser()
    config.read('C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\config.ini')
    file_path = config['Paths']['test_file_path']
    path = config['Paths']['train_data_path']
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model.eval()
    model.to(device)

    model_eval, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model_eval.eval()

    descriptions = []
    scores = []
    image_names = []

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Remove any leading or trailing whitespaces and newline characters
            image_name = line.strip()
            image_name = path + image_name
            print(image_name)
            image_names.append(image_name)
            image_or = Image.open(image_name).convert('RGB')

            image = transform(image_or).unsqueeze(0).to(device)
            with torch.no_grad():
                generated = model.generate(image)

            description = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")[:-2]
            print(description)
            descriptions.append(description)

            model.to('cpu')
            model_eval.to(device)

            img = preprocess(image_or).unsqueeze(0).to(device)
            text_tokens = tokenizer.tokenize(description).to(device)

            with torch.no_grad():
                image_features = model_eval.encode_image(img)
                text_features = model_eval.encode_text(text_tokens)

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # Calculate the cosine similarity to get the CLIP score
            clip_score = torch.matmul(image_features, text_features.T).item()
            print(clip_score)
            scores.append(clip_score)

            model_eval.to('cpu')
            model.to(device)

    image_test = Image.open(image_names[22]).convert('RGB')
    plt.imshow(image_test)
    plt.title(descriptions[22], size=15)
    plt.axis('off')
    plt.show()

    print(np.mean(scores))

    with open('../robaccia/Captions/log_captions.txt', 'w') as file:
        for i, j, k in zip(image_names, descriptions, scores):
            file.write(f"{i} --- {j} --- {k}\n")