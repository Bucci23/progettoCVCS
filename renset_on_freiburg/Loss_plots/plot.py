import matplotlib.pyplot as plt

epochs = []
losses = []

with open('logsRESNET512.txt', 'r') as file:
    for line in file:
        parts = line.split(':')
        epoch = int(parts[0].strip().replace('epoch', ''))
        print(epoch)
        loss = float(parts[1].split()[4].strip())

        epochs.append(epoch)
        losses.append(loss)

plt.plot(epochs, losses)
plt.xlabel('Epoch')
plt.ylabel('Global Avg Loss')
plt.title('ResNet')
plt.show()
