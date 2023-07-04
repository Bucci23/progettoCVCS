import skudataset
import multiprocessing
import math
import plot_data
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights, \
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, \
    fasterrcnn_mobilenet_v3_large_fpn, \
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, \
    fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights
from engine import train_one_epoch, evaluate
import utils as utils
import transforms as T
import load_sku
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
from engine import train_one_epoch, evaluate
import utils
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import gc
gc.collect()
torch.cuda.empty_cache()

def get_object_detection_model(num_classes  , retina=False):
    # weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    # weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    # weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    # weights = None
    # load a model pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    if retina:
        model = retinanet_resnet50_fpn_v2(weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
        # in_features = model.head.classification_head.conv[0].in_channels
        # num_anchors = model.head.classification_head.num_anchors
        # model.head.classification_head.num_classes = num_classes
        # cls_logits = torch.nn.Conv2d(in_features, num_anchors * num_classes, kernel_size = 3, stride=1, padding=1)
        # torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
        # torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
        # # assign cls head to model
        # model.head.classification_head.cls_logits = cls_logits
    else:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model



base_dir = '/work/cvcs_2023_group12/SKU110K_fixed/SKU110K_fixed/'
img_dir = '/work/cvcs_2023_group12/SKU110K_fixed/SKU110K_fixed/images'

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # loading the dataset

    dataset = skudataset.SkuDataset(base_dir, img_dir, 1024, 1024, transforms=skudataset.get_transform(train=True), size='train')
    dataset_test = skudataset.SkuDataset(base_dir, img_dir, 1024, 1024, transforms=skudataset.get_transform(train=False), size = 'val')
    print('loaded the dataset')
    # split the dataset in train and test set
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=6, shuffle=True,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=6, shuffle=False,
        collate_fn=utils.collate_fn)
    print('created data loaders')
    print('this training set has length = ', len(dataset))
    print('the test set has length = ', len(dataset_test))
    print('sample img:')
    im, trg = dataset[27]
    print(trg)
    plot_data.plot_img_bbox(skudataset.torch_to_pil(im), trg, 1024, 1024)
    # TRAINING:
    num_classes = 2

    # Use a pre-trained Faster R-CNN model with a ResNet-50 backbone
    model = get_object_detection_model(num_classes, retina=False)
    # Set the model to training mode
    model.train()

    # Define the device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    train = True
    if device != torch.device('cuda'):
        train = False
    print(device)
    # Define hyperparameters
    num_epochs = 100
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, eps=1e-5, cooldown=0, min_lr=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    # Training loop
    train = True
    if(train):
        print('Training started...')
        for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
            # update the learning rate
            file = open('newlogs.txt', 'a')
            loss = metrics.meters['loss'].global_avg
            file.write(f"epoch{epoch}:{metrics}")
            scheduler.step(loss)
            lr = optimizer.param_groups[0]['lr']
            file.write(f'new learning rate = {lr}')
            file.close()
            # evaluate on the test dataset
            if epoch % 5 == 0:
                evaluate(model, data_loader_test, device=device)
            torch.save(model, f'1024_model_epoch{epoch+1}.pth')
            print('Model saved!')