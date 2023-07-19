import os
import configparser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mpmath
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

    image_or = Image.open(
        "C:\\Users\\pavon\\Downloads\\Grocery_products\\Testing/store2/images/104.jpg").convert('RGB')

    image = transform(image_or).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = model.generate(image)

    description = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")[:-2]
    print(description)

    plt.imshow(image_or)
    plt.title(description, size=15)
    plt.axis('off')
    plt.show()

    model_eval, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai')
    model_eval.eval()
    model_eval

    image_or = Image.open(
        "C:\\Users\\pavon\\Downloads\\Grocery_products\\Testing/store2/images/104.jpg")

    img = preprocess(image_or).unsqueeze(0)

    """
    normalize = transforms.Compose([
        transforms.ToTensor()
    ])"""


    text_tokens = tokenizer.tokenize(description)

    with torch.no_grad():
        image_features = model_eval.encode_image(img)
        text_features = model_eval.encode_text(text_tokens)

    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Calculate the cosine similarity to get the CLIP score
    clip_score = torch.matmul(image_features, text_features.T).item()
    print(clip_score)

    #print(image_features.shape, text_features.shape)
    #print(img.squeeze(0).shape)
    #image_or = normalize(image_or)
    #print(image_or.shape)
"""
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    score = metric(image_or, description)
    score = metric(image_features, text_features)
    print(score.detach())

"""


