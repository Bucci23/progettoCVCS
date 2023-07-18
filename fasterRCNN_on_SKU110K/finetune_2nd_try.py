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
def get_object_detection_model(num_classes):
    # weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    # weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT

    # load a model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    # model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=weights)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')


# Send train=True from training transforms and False for val/test transforms
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(True),
            # ToTensorV2 converts image to pytorch tensor without div by 255

            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'albumentations', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'albumentations', 'label_fields': ['labels']})


if __name__ == '__main__':
    multiprocessing.freeze_support()
    # loading the dataset
    base_dir = 'C:\\Users\\david\\Desktop\\dataset\\sku110k\\SKU110K_fixed.tar\\SKU110K_fixed\\SKU110K_fixed\\'
    img_dir = 'C:\\Users\\david\\Desktop\\dataset\\sku110k\\SKU110K_fixed.tar\\SKU110K_fixed\\SKU110K_fixed\\images\\'
    dataset = skudataset.SkuDataset(base_dir, img_dir, 300, 300, transforms=get_transform(train=True))
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    # train test split
    test_split = 0.2
    tsize = int(len(dataset) * test_split)
    dataset_test = torch.utils.data.Subset(dataset, indices[-tsize:])
    dataset = torch.utils.data.Subset(dataset, indices[:-tsize])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=8, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=8, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)
    print('this training set has length = ', len(dataset))
    print('the test set has length = ', len(dataset_test))
    print('sample img')
    im, trg = dataset[27]
    print(trg)
    plot_data.plot_img_bbox(torch_to_pil(im), trg)
    # TRAINING:

    # to train on gpu if selected.
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')

    num_classes = 2

    # get the model using our helper function
    model = get_object_detection_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    # training for 10 epochs
    num_epochs = 30

    for epoch in range(num_epochs):
        # training for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        torch.save(model.state_dict(), 'model_weights.pth')
        evaluate(model, data_loader_test, device=device)
    img, trg = dataset[34]
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])[0]
    print(prediction)
    plot_data.plot_img_bbox(prediction)
