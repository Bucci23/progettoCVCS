import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
import cv2
from finetune import get_object_detection_model


#weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
#weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
#weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
# model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
# model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)


num_classes = 81

# get the model using our helper function
#model = get_object_detection_model(num_classes)

#model.load_state_dict(torch.load('model_weights.pth'))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

model.eval()

a = "C:\\Users\\pavon\\Downloads\\Grocery_products\\Testing\\store5\\images\\37.jpg"
img = cv2.imread(a)
# Convert the image to a PyTorch tensor
img = torch.from_numpy(img).permute(2, 0, 1).float().to(device)
#img = torch.from_numpy(img).float().to(device)
# Get the predictions
with torch.no_grad():
    pred = model([img])
    print(pred)

# Get the bounding boxes, scores, and labels from the predictions
boxes = pred[0]['boxes'].cpu().numpy()
scores = pred[0]['scores'].cpu().numpy()
labels = pred[0]['labels'].cpu().numpy()

# Show the image and the bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(np.transpose(img.cpu().numpy(), (1, 2, 0)))
#ax.imshow(img.cpu().numpy())

for box, score, label in zip(boxes, scores, labels):
    if score > 0.5: # threshold for detection
        xmin, ymin, xmax, ymax = box
        width = xmax - xmin
        height = ymax - ymin
        rect = plt.Rectangle((xmin, ymin), width, height, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin, f'{score:.2f}', fontsize=8, color='white', bbox=dict(facecolor='red', alpha=0.5))
        ax.text(xmin, ymin + 15, f'{label}', fontsize=8, color='white', bbox=dict(facecolor='red', alpha=0.5))

plt.show()