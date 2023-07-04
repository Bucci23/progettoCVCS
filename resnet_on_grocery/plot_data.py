import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_bbox(img,class_index, w=512, h=512):
    fig, a  = plt.subplots(1, 1)
    a.imshow(img)
    a.set_title(class_index)