import os
import numpy as np
import cv2
from torch.utils.data import Dataset
import torch
import utils as utils
import load_sku


class SkuDataset(Dataset):
    def __init__(self, base_dir,img_dir, width, height, transforms=None):
        self.transforms = transforms
        self.height = height
        self.classes = [0]
        self.width = width
        self.annotations = load_sku.read_csv_annotations(os.path.join(base_dir, 'annotations\\annotations_val.csv'))
        self.imgs = list(self.annotations.keys())
        self.img_dir = img_dir
        print(len(self.classes))

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.imgs[idx])
        print(img_name)
        img = cv2.imread(img_name)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        # diving by 255
        img_res /= 255.0
        boxes = []
        labels = [0 for i in range(len(self.annotations[self.imgs[idx]]))]
        relative_boxes = []
        wt = img.shape[1]
        ht = img.shape[0]
        specific_annot = self.annotations[self.imgs[idx]]
        for obj in specific_annot:
            xmin = obj[0]
            ymin = obj[1]
            xmax = obj[2]
            ymax = obj[3]
            rel_xmin = xmin/wt
            rel_xmax = xmax/wt
            rel_ymin = ymin/ht
            rel_ymax = ymax/ht
            if xmin < 0:
                xmin = 0
                rel_xmin = 0
            if ymin < 0:
                ymin = 0
                rel_ymin = 0
            if xmax < 0:
                xmax = 0
                rel_xmax = 0
            if ymax < 0:
                ymax = 0
                rel_ymax = 0
            rel_ymax = 1 if rel_ymax > 1 else rel_ymax
            rel_ymin = 1 if rel_ymin > 1 else rel_ymin
            rel_xmax = 1 if rel_xmax > 1 else rel_xmax
            rel_xmin = 1 if rel_xmin > 1 else rel_xmin

            boxes.append([xmin, ymin, xmax, ymax])
            relative_boxes.append([rel_xmin, rel_ymin, rel_xmax, rel_ymax])

        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        # getting the areas of the boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # suppose all instances are not crowd
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {"boxes": relative_boxes, "labels": labels, "area": area, "iscrowd": iscrowd}
        # image_id
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms:
            try:
                sample = self.transforms(image=img_res,
                                         bboxes=relative_boxes,
                                         labels=labels)
                img_res = sample['image']
                target['boxes'] = torch.Tensor(sample['bboxes'])
            except:
                print('rotto a causa di ', img_name, ' con BBOX: ', target['boxes'])
        return img_res, target

    def __len__(self):
        return len(self.imgs)

