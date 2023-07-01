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
from open_clip import tokenizer
import glob

from Classification.preparation import FileOperations

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model.eval()
    model.to(device)


    # IMAGE PREPROCESSING
    # We resize the input images and center-crop them to conform with the image resolution that the model expects.
    # Before doing so, we will normalize the pixel intensity using the dataset mean and standard deviation.

    # TEXT PREPROCESSING
    # We use a case-insensitive tokenizer
    #  the outputs are padded to become 77 tokens long, which is what the CLIP models expects

    test_path = "C:\\Users\\pavon\\Downloads\\freiburg-groceries.v10-original.clip\\test"
    test_file = "C:\\Users\\pavon\\Downloads\\freiburg-groceries.v10-original.clip\\test\\_tokenization_better.txt"

    # Create an instance of the class
    file_ops = FileOperations(test_path)

    # Get directory names
    directory_names = file_ops.get_directory_names()
    print(directory_names)
    classes = directory_names

    # Read file lines
    file_path = test_file
    lines = file_ops.read_file_lines(file_path)
    for line in lines:
        print(line.strip())

    # Create the descriptions dictionary
    descriptions = file_ops.create_description_dict(file_path)
    print(descriptions)
    print(descriptions.values())

    text_tokens = tokenizer.tokenize([desc for desc in descriptions.values()]).to(device)
    print(text_tokens)

    label = 0
    correct = []

    for cls in classes:
        label +=1
        class_correct = []
        test_imgs = glob.glob(test_path + '\\' + cls + '/*.jpg')
        print(test_imgs)
        for img in test_imgs:
            image = Image.open(img)
            image = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = model.encode_image(image).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features = model.encode_text(text_tokens).float()
                text_features /= text_features.norm(dim=-1, keepdim=True)

            # classify images using the cosine similarity (times 100) as the logits to the softmax operation

            #text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            #top_probs, top_labels = text_probs.to('cpu').topk(5, dim=-1)
            #print(top_probs, top_labels)
            #pred = 0

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            values, indices = similarity[0].topk(1)
            print(f"{cls}: {values[0].item():.2f}")
            print(indices[0].item(), label)

            if indices[0].item() == label:
                correct.append(1)
                class_correct.append(1)
            else:
                correct.append(0)
                class_correct.append(0)

        print('accuracy on class ' + cls + ' is :' + str(sum(class_correct) / len(class_correct)))
    print('accuracy on all is : ' + str(sum(correct) / len(correct)))




    image_or = Image.open('C:\\Users\\pavon\\Downloads\\freiburg-groceries.v10-original.clip\\test\\beans\\BEANS0005_png.rf.1ac6a53cba0ef42961796f2073839788.jpg')
    image = preprocess(image_or).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image).float()
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    # classify images using the cosine similarity (times 100) as the logits to the softmax operation

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.to('cpu').topk(3, dim=-1)
    plt.figure(figsize=(16, 16))
    plt.imshow(image_or)
    plt.axis("off")

    plt.show()

    print(top_probs, top_labels)