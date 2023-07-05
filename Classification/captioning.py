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
        'C:\\Users\\pavon\\Downloads\\freiburg-groceries.v10-original.clip'
        '\\test\\milk_multiple\\MILK0148_png.rf.945ff15feca2beb189ffddffc6357721.jpg').convert('RGB')

    image_or = Image.open(
        'C:\\Users\\pavon\\Documents\\progettoCVCSv1.0\\data\\uniform\\Foto7.jpg').convert('RGB')

    image_or = Image.open(
        'C:\\Users\\pavon\\Downloads\\Grocery_products'
        '\\Testing\\store1\\images\\10.jpg').convert('RGB')


    image = transform(image_or).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = model.generate(image)


    description = open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")[:-2]
    print(description)

    plt.imshow(image_or)
    plt.title(description, size=15)
    plt.axis('off')
    plt.show()



