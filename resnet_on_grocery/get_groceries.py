import torch
import cv2
import os
import load_groceries
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as torchtrans
import numpy as np
import plot_data
base_dir = '/work/cvcs_2023_group12/Grocery_products/Training'
class GroceryDataset(torch.utils.data.Dataset): # inheritin from Dataset class
    def __init__(self, root_dir="", w=512, h=512, transform=None):
        self.annotation_dict, self.class_dict = load_groceries.get_data_dict(root_dir)
        self.transform = transform
        self.base_dir = root_dir
        self.w = w
        self.h = h
        self.num_classes = len(self.class_dict)
        self.transform = transform
    def __len__(self):
        return len(self.annotation_dict) # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        #get list of images from the keys of the dictionary:
        list_of_images = list(self.annotation_dict.keys())
        image_path = os.path.join(self.base_dir, list_of_images[idx]) #use image path column (index = 1) in csv file
        image = cv2.imread(image_path) # read image by cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) # convert from BGR to RGB
        img_res = cv2.resize(image, [self.w, self.h], cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        # Find the class name and index of the image:
        class_index = self.annotation_dict[list_of_images[idx]] # use class name column (index = 2) in csv file
        #Find the corresponding class name from the keys of the class_dict:
        class_name = list(self.class_dict.keys())[class_index]
        if self.transform:
            try:
                smaple = self.transform(image=img_res)
                img_res = smaple["image"]
            except:
                print("Error in transform caused by image: ", image_path)
        return img_res, class_index
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(True),  # horizontal flip    
            A.RandomRotate90(p=0.7),  # rotation of 90°    
            A.ChannelShuffle(p=0.7),  # mixing RGB channels
            A.RandomShadow(p=0.7), # Generating random shadows    
            ToTensorV2(p=1.0)
        ])
    else:
        return A.Compose([
            ToTensorV2(p=1.0)
        ])

def torch_to_pil(img):
    return torchtrans.ToPILImage()(img).convert('RGB')

if __name__ == '__main__':
    dataset = GroceryDataset(base_dir, 512, 512, transform=get_transform(train=True))
    print(len(dataset))
    img, class_index = dataset[0]
    plot_data.plot_img_bbox(torch_to_pil(img), class_index)
