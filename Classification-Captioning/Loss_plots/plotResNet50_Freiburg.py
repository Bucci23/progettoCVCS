# PLOT OF THE LOSS FUNCTION (RESNET50 ON FREIBURG DATASET)

import matplotlib.pyplot as plt

loss_values = [1.5646, 0.5577, 0.2312, 0.1233, 0.0588, 0.0620, 0.0455, 0.0312, 0.0197, 0.0145]
epochs = range(1, len(loss_values) + 1)

plt.plot(epochs, loss_values, 'b-o')
plt.title('Loss function plot (ResNet50 on Freigburg dataset)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)

plt.show()