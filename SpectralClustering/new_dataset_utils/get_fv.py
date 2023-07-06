import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
base_dir = 'C:\\Users\\david\\Desktop\\dataset\\sku110k\\SKU110K_fixed.tar\\SKU110K_fixed\\SKU110K_fixed\\images'
def get_X():
    dataset = pd.read_csv('new_dataset_utils\\positions.csv')
    annotation_dict = {}
    grouped_by_name = dataset.groupby('name')
    for name in grouped_by_name.groups.keys():
        annotation_dict[name] = grouped_by_name.get_group(name).drop(['name', 'centerX'], axis=1)
    max_length = 0
    for name, fv in annotation_dict.items():
        if(max_length < fv.shape[0]):
            max_length = fv.shape[0]
    print(max_length)

    for name, fv in annotation_dict.items():
        out = np.zeros((1, max_length))
        out[0, 0:fv.shape[0]] = fv['centerY'].to_numpy().reshape(1,-1)
        annotation_dict[name] = out
    #print(annotation_dict)
    #Create a tensor with all the feature vectors
    X_dataset = np.zeros((len(annotation_dict), max_length))
    for image_index in range(len(annotation_dict)):
        image_name = list(annotation_dict.keys())[image_index]
        image_data = annotation_dict[image_name]
        X_dataset[image_index, 0:image_data.shape[1]] = image_data
    return X_dataset

if __name__ == '__main__':
    x_dataset = get_X()
    #load and show all the images in the 'names.csv' file from the base_dir
    names = pd.read_csv('new_dataset_utils\\names.csv')
    print(names)
    for i in range(len(names)):
        name = names['name'][i]
        print(name)
        image_path = os.path.join(base_dir, name)
        image = plt.imread(image_path)
        #Set the title to name
        plt.title(name)
        plt.imshow(image)
        plt.show()
