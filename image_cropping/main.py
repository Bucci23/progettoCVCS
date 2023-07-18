from SingleObjectDetectionCanny import single_object_detect
import accuracy
import numpy as np
import matplotlib.pyplot as plt
import json
import os

input_dir = 'data/uniform'
output_dir = 'out/uniform'
mask_dir = 'robaccia/edges/'
intermediate_result = 'robaccia/intermediate'
grabCut_images = 'robaccia/grabCut'
predictions_dir = 'predictions/'


def thl_cycle(start, finish, step, thh_factor=2):
    for thl in range(start, finish + 1, step):
        predictions = single_object_detect(input_dir, output_dir, mask_dir, intermediate_result, grabCut_images, thl,
                                           thh=thh_factor * thl)
        with open(f"{predictions_dir}pred_thl={thl}.json", "w") as write_file:
            json.dump(predictions, write_file)
        print(thl, 'done')


def best_thresh(start, finish, step):
    best_iou = 0
    best_predictions = {}
    best_threshold = 0
    means = np.zeros(int((finish - start)/step)+1)
    variances = np.zeros(int((finish - start)/step)+1)
    for thl in range(start, finish + 1, step):
        with open(f"{predictions_dir}/pred_thl={thl}.json", "r") as read_file:
            predictions = json.load(read_file)
        iou_array = accuracy.iou(predictions, 'dataset_annotations.csv')
        mean = np.mean(iou_array)
        var = np.var(iou_array)
        means[int((thl - start) / step)] = mean
        variances[int((thl - start) / step)] = var
        print(mean, thl)
        if mean > best_iou:
            best_iou = mean
            best_threshold = thl
    return best_threshold, means, variances


def means_vars_plot(mean, var, start, finish, step, name):
    plt.errorbar(np.arange(mean.size), mean, yerr=np.sqrt(var), fmt='o')
    # Customize plot
    plt.xticks(np.arange(mean.size), [i for i in range(start, finish+1, step)])
    plt.xlabel('THL values')
    plt.ylabel('mean IoU')
    plt.title('IOU Means and variances')
    plt.savefig(name, dpi=300)
    # Show plot
    plt.show()


if __name__ == '__main__':
    # Plot means with error bars
    #best_thl, means_to_plot, vars_to_plot = best_thresh(50, 125, 5)
    #means_vars_plot(means_to_plot, vars_to_plot, 50, 125, 5, 'plotthl2thh')
    factors = {}
    best_factor = 0
    best_iou = 0
    for factor in np.arange(1.25, 2.51, 0.25):
        predictions_dir = f'predictions{int(100 * factor)}/'
        if not os.path.exists(predictions_dir):
            os.makedirs(predictions_dir)
        thl_cycle(50, int(255 / factor), 5, factor)
        best_th, means, var = best_thresh(50, int(255 / factor), 5)
        factors[factor] = [best_th, means[int((best_th-50)/5)]]
        if factors[factor][1] > best_iou:
            best_iou = factors[factor][1]
            best_factor = factor
        means_vars_plot(means, var, 50, int(255 / factor), 5, f'plotthl{int(100 * factor)}thh.png')
    with open(f"factors.json", "w") as write_file:
        json.dump(factors, write_file)
# with thh = 2xthl the best is 110 ->0.84
