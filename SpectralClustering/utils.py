import os
import csv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


if __name__ == '__main__':
    folder_path = "images"  # Percorso della cartella

    # Ottieni la lista dei file nella cartella
    file_names = os.listdir(folder_path)

    nomi = []

    # Stampa i nomi dei file
    for file_name in file_names:
        print(file_name)
        nomi.append(file_name)

    print(nomi)

    folder_path = "test"  # Percorso della cartella

    # Ottieni la lista dei file nella cartella
    file_names = os.listdir(folder_path)

    nomi = []

    for file_name in file_names:
        print(file_name)
        nomi.append(file_name)

    print(nomi)


# Dizionario per tenere traccia delle bounding box per ogni nome di file
bounding_boxes = {}
pre = 'images/'

# Apri il file CSV
with open('spectral_clustering.csv', 'r') as file:
    csv_reader = csv.reader(file)

    # Salta la riga dell'intestazione se esiste
    #next(csv_reader)

    # Itera su ogni riga nel file CSV
    for row in csv_reader:
        if row[0] == 'train_3410.jpg':
            pre = 'test/'
        image_path = pre + row[0]

        x1, y1, x2, y2 = map(int, row[1:5])
        class_label = float(row[5])

        # Aggiungi la bounding box alla lista corrispondente al nome del file
        if image_path not in bounding_boxes:
            bounding_boxes[image_path] = []

        bounding_boxes[image_path].append((x1, y1, x2, y2, class_label))

print(bounding_boxes)

# Itera sui nomi dei file e le relative bounding box
for image_path, bboxes in bounding_boxes.items():
    # Carica l'immagine
    image = Image.open(image_path)

    # Crea una figura e un set di assi per l'immagine corrente
    fig, ax = plt.subplots()

    # Mostra l'immagine
    ax.imshow(image)

    # Itera sulle bounding box e plottale
    for bbox in bboxes:
        x1, y1, x2, y2, class_label = bbox

        # Colori diversi per classi pari o dispari
        if class_label % 2 == 0:
            rect_color = 'red'
        else:
            rect_color = 'blue'

        # Crea una patch rettangolare per la bounding box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=rect_color, facecolor='none')

        # Aggiungi la patch rettangolare agli assi
        ax.add_patch(rect)

    # Mostra il grafico per l'immagine corrente
    plt.show()
