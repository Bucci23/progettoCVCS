import csv
import cv2
import json
import numpy as np


def load_box_annotations(annotation_filename):
    # Open the CSV file
    csvfile = open(annotation_filename, newline='')
    # Create a CSV reader object
    reader = csv.reader(csvfile, delimiter=',')

    # Skip the header row
    next(reader)

    # Loop over the remaining rows and store the bounding boxes in a list
    ground_truth_boxes = []
    gt_map = {}
    for row in reader:
        gt_attr = json.loads(row[5])
        gt_map[row[0]] = [gt_attr['x'], gt_attr['y'], gt_attr['width'], gt_attr['height']]
        '''x, y, w, h = map(int, row[1:])
        ground_truth_boxes.append((x, y, w, h))'''
    return gt_map


def iou(predictions, annotation_filename):
    gt = load_box_annotations(annotation_filename)
    print(gt)
    print(predictions)
    ious = []
    for name, box in gt.items():
        if name in predictions:
            box2 = predictions[name]
            x1 = max(box[0], box2[0])
            y1 = max(box[1], box2[1])
            x2 = min(box[0] + box[2], box2[0] + box2[2])
            y2 = min(box[1] + box[3], box2[1] + box2[3])
            intersection = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the area of each bounding box
            area1 = box[2] * box[3]
            area2 = box2[2] * box2[3]

        # Calculate the union area
            union = area1 + area2 - intersection

        # Calculate the IoU
            result = intersection / union if union > 0 else 0

            ious.append(result)
    return np.array(ious)
