import os
def load_img(rood_dir):
    filename = os.path.join(rood_dir, 'TrainingFiles.txt')
    dataset = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            #change the '/' in '\\' inside the line:
            line = line.replace('/', '\\')
            #remove the whitespaces
            line = line.strip()
            fullname = os.path.join(rood_dir, line)
            dataset[fullname] = os.path.dirname(line)

    return dataset
def load_basenames(root_dir):
    filename = os.path.join(root_dir, 'TrainingFiles.txt')
    basenames = {}
    with open(filename) as f:
        lines = f.readlines()
        for line in lines:
            #change the '/' in '\\' inside the line:
            line = line.replace('/', '\\')
            #remove the whitespaces
            line = line.strip()
            fullname = os.path.join(root_dir, line)
            basenames[fullname] = os.path.basename(line)
    return basenames
