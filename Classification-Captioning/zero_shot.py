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
    val_path = "C:\\Users\\pavon\\Downloads\\freiburg-groceries.v10-original.clip\\valid"
    test_file = "C:\\Users\\pavon\\Downloads\\freiburg-groceries.v10-original.clip\\test\\_tokenization_best.txt"

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

    unique_values = []
    [unique_values.append(value) for value in descriptions.values() if value not in unique_values]
    print(unique_values)

    text_tokens = tokenizer.tokenize([desc for desc in unique_values]).to(device)
    print(text_tokens[0], text_tokens[12])

    label = [i for i in itertools.chain.from_iterable(zip(range(1, 26), range(1, 26)))]
    correct = []
    i = 0
    true_labels = []
    predicted_labels = []

    for cls in classes:
        class_correct = []
        test_imgs = glob.glob(test_path + '\\' + cls + '/*.jpg')
        test_imgs = test_imgs + glob.glob(val_path + '\\' + cls + '/*.jpg')
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
            #print(similarity[0].topk(10))
            #print(f"{cls}: {values[0].item():.2f}")
            #print(indices[0].item(), label[i])

            if indices[0].item() + 1 == label[i]:
                correct.append(1)
                class_correct.append(1)
                true_labels.append(label[i])
                predicted_labels.append(indices[0].item() + 1)
            else:
                correct.append(0)
                class_correct.append(0)
                true_labels.append(label[i])
                predicted_labels.append(indices[0].item() + 1)

        #print('accuracy on class ' + cls + ' is :' + str(sum(class_correct) / len(class_correct) + mpmath.eps))
        i = i+1
    #print('accuracy on all is : ' + str(sum(correct) / len(correct)))
    accuracy = metrics.accuracy_score(true_labels, predicted_labels)
    precision = metrics.precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
    recall = metrics.recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
    f1_score = metrics.f1_score(true_labels, predicted_labels, average='macro', zero_division=0)

    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1_score)
    cm = ConfusionMatrixDisplay.from_predictions(true_labels, predicted_labels)
    cm.plot()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

"""
    image_or = Image.open('C:\\Users\\pavon\\Downloads\\freiburg-groceries.v10-original.clip\\test\\water_multiple\\WATER0244_png.rf.563b8a40887995be8f305feed0c7b1c3.jpg')
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

    print(top_probs, top_labels)"""