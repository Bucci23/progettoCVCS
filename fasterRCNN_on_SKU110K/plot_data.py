import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_img_bbox(img, target, w=512, h=512, relative=True):
    # plot the image and bboxes
    # Bounding boxes are defined as follows: x-min y-min width height
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(img)
    for box in (target['boxes']):
        if relative:
            x, y, width, height = box[0]*w, box[1]*h, (box[2] - box[0])*w, (box[3] - box[1])*h
        else:
            x, y, width, height = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect = patches.Rectangle((x, y),
                                 width, height,
                                 linewidth=2,
                                 edgecolor='r',
                                 facecolor='none')

        # Draw the bounding box on top of the image
        a.add_patch(rect)
    plt.show()
