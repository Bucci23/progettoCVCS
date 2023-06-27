import os

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