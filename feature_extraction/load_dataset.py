import os
import json
def load_img(rood_dir):
    filename = os.path.join(rood_dir, 'TrainingFiles.txt')
    dataset = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            #change the '/' in '\\' inside the line:
            line = line.replace('/', '\\')
            #remove the whitespaces
            line = line.strip()
            fullname = os.path.join(rood_dir, line)
            dataset[fullname] = os.path.dirname(line)

    return dataset
def load_basenames(root_dir):
    filename = os.path.join(root_dir, 'TrainingFiles.txt')
    basenames = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            #change the '/' in '\\' inside the line:
            line = line.replace('/', '\\')
            #remove the whitespaces
            line = line.strip()
            fullname = os.path.join(root_dir, line)
            basenames[fullname] = os.path.basename(line)
    return basenames

def crop_image(img, bbox):
    #img is rgb, bbox is [x1, y1, w, h]
    x1, y1, w, h = bbox
    x2 = x1 + w
    y2 = y1 + h
    return img[:, y1:y2, x1:x2]

def load_json_annotations(filename):
    with open(filename) as f:
        annotations = json.load(f)
    return annotations

def load_json_annotations_testing(filename):
    with open(filename) as f:
        annotations = json.load(f)
    dictionary = {}
    for i in annotations.values():
        dictionary[i['filename']] = [i['regions'][0]['shape_attributes']['x'], i['regions'][0]['shape_attributes']['y'], i['regions'][0]['shape_attributes']['width'], i['regions'][0]['shape_attributes']['height']]
    return dictionary