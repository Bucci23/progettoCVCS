import os
import cv2


base_dir = 'C:\\Users\david\\Desktop\\Scuola dav\\SKU110K_fixed\\Grocery_products\\Training'
#I want to explore the directories from base_dir to get a list with the names of all the images:
def load_image_list(base_dir):
    list_of_images = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.endswith((".jpg")):
                #append to the list the relative name of the file with respect to the base_dir:
                list_of_images.append(os.path.relpath(os.path.join(root, name), base_dir))
            else:
                continue
    return list_of_images

def extract_dir(list_of_images):
    #I want to remove the name of the file and keep only the name of the directory:
    list_of_directories = []
    for image in list_of_images:
        list_of_directories.append(os.path.dirname(image))
    return list_of_directories

def map_to_numeric(list_of_directories):
    #I want to map the names of the directories to a numeric value:
    #I create a dictionary:
    dict_of_directories = {}
    for i in range(len(list_of_directories)):
        if list_of_directories[i] not in dict_of_directories:
            dict_of_directories[list_of_directories[i]] = len(dict_of_directories)
        else:
            continue
    return dict_of_directories
def get_numeric_class_dict(list_of_images, list_of_directories, dict_of_directories):\
    #I want to create a dictionary that maps the name of the image to the numeric value of the directory:
    dict_of_images = {}
    for i in range(len(list_of_images)):
        dict_of_images[list_of_images[i]] = dict_of_directories[list_of_directories[i]]
    return dict_of_images

def get_data_dict(base_dir):
    #I want to create a dictionary that maps the name of the image to the numeric value of the directory:
    list_of_images = load_image_list(base_dir)
    list_of_directories = extract_dir(list_of_images)
    dict_of_directories = map_to_numeric(list_of_directories)
    dict_of_images = get_numeric_class_dict(list_of_images, list_of_directories, dict_of_directories)
    return dict_of_images, dict_of_directories



#main to test the function:
# if __name__ == '__main__':
#     dict_of_images, dict_of_directories = get_data_dict(base_dir)
#     print(len(dict_of_images))