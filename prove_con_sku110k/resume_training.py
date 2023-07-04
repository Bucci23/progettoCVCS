import skudataset
import multiprocessing
import plot_data
import torch
import torchvision
from torchvision import transforms as torchtrans
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_ResNet50_FPN_V2_Weights, \
    fasterrcnn_resnet50_fpn_v2, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights, \
    fasterrcnn_mobilenet_v3_large_fpn, \
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights, \
    fasterrcnn_mobilenet_v3_large_320_fpn

from engine import train_one_epoch, evaluate
import utils as utils
import transforms as T
import load_sku
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader
import cv2
from engine import train_one_epoch, evaluate
import utils

base_dir = '/work/cvcs_2023_group12/SKU110K_fixed/SKU110K_fixed/'
img_dir = '/work/cvcs_2023_group12/SKU110K_fixed/SKU110K_fixed/images'
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import gc
gc.collect()
torch.cuda.empty_cache()
if __name__ == '__main__':
    multiprocessing.freeze_support()
    # loading the dataset
    print('loading dataset')
    dataset = skudataset.SkuDataset(base_dir, img_dir, 1024, 1024, transforms=skudataset.get_transform(train=True), size='train')
    dataset_test = skudataset.SkuDataset(base_dir, img_dir, 1024, 1024, transforms=skudataset.get_transform(train=False), size = 'val')
    # define training and validation data loaders
    print('creating data loaders')
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=6, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=6, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)
    print('this training set has length = ', len(dataset))
    print('the test set has length = ', len(dataset_test))
    # print('sample img:')
    # im, trg = dataset[27]
    # print(trg)
    # plot_data.plot_img_bbox(skudataset.torch_to_pil(im), trg, 512, 512)
    # TRAINING:
    model = torch.load('1024_model_epoch10.pth')
    model.train()

    # Define the device (CPU or GPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    train = True
    if device != 'cuda':
        print(device)
        train = False
    # Define hyperparameters
    num_epochs = 100

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.00005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, eps=1e-5, cooldown=0, min_lr=0)
    # Training loop
    train = True
    if(train):
        print('Training started...')
        for epoch in range(10, num_epochs, 1):
            # train for one epoch, printing every 10 iterations
            metrics = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
            file = open('newlogs.txt', 'a')
            loss = metrics.meters['loss'].global_avg
            file.write(f"epoch{epoch}:{metrics}\n")
            # update the learning rate
            scheduler.step(loss)
            # evaluate on the test dataset
            lr = optimizer.param_groups[0]['lr']
            file.write(f'new learning rate = {lr}\n')
            file.close()
            torch.save(model, f'1024model_epoch_{epoch+1}.pth')
            print('Model saved!')
            if epoch % 5 == 0:
                evaluate(model, data_loader_test, device=device)
            
            