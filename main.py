from SingleObjectDetectionCanny import single_object_detect
import accuracy
import numpy as np

input_dir = 'data/'
output_dir = 'out/'
mask_dir = 'robaccia/edges/'
intermediate_result = 'robaccia/intermediate'
grabCut_images = 'robaccia/grabCut'

if __name__ == '__main__':
    predictions = single_object_detect(input_dir, output_dir, mask_dir, intermediate_result, grabCut_images)
    iou_array = accuracy.iou(predictions, 'dataset_annotations.csv')
    print(iou_array)
    print(np.mean(iou_array))