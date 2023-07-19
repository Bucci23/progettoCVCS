import matplotlib.pyplot as plt

epochs = []

with open('logsRESNET1024.txt', 'r') as file:
    for line in file:
        parts = line.split(':')
        epoch = int(parts[0].replace('epoch', '').strip())
        epochs.append(epoch)

print('Epochs:', epochs)

loss = [2.1356, 1.1882, 1.0395, 0.9644, 0.9211, 0.8908, 0.8657, 0.8479, 0.8317, 0.8182]
loss_classifier = [0.4457,
                   0.2923,
                   0.2558,
                   0.2361,
                   0.2246,
                   0.2165,
                   0.2100,
                   0.2047,
                   0.2006,
                   0.1966]
loss_box_reg = [
    0.5568,
    0.5112,
    0.4546,
    0.4284,
    0.4141,
    0.4037,
    0.3962,
    0.3899,
    0.3845,
    0.3802
]

loss_objectness = [0.9064,
                   0.2226,
                   0.1862,
                   0.1673,
                   0.1560,
                   0.1487,
                   0.1413,
                   0.1374,
                   0.1333,
                   0.1294]

loss_rpn = [0.2267,
            0.1621,
            0.1429,
            0.1326,
            0.1264,
            0.1219,
            0.1182,
            0.1158,
            0.1133,
            0.1120]

plt.plot(epochs, loss, label='Loss')
plt.plot(epochs, loss_classifier, label='Loss Classifier')
plt.plot(epochs, loss_box_reg, label='Loss Box Reg')
plt.plot(epochs, loss_objectness, label='Loss Objectness')
plt.plot(epochs, loss_rpn, label='Loss RPN')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Losses Over Epochs')
plt.legend()

plt.show()
