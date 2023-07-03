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
        '\\test\\beans\\BEANS0020_png.rf.2af940f2422d01c4bd5c4afdb2c54698.jpg').convert('RGB')
    image = transform(image_or).unsqueeze(0).to(device)
    with torch.no_grad():
        generated = model.generate(image)

    print(open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")[:-2])
