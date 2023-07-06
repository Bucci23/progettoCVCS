import pandas as pd
dataset = pd.read_csv('new_dataset_utils\\annotations_val.csv')
dataset['centerX'] = dataset['xmax']+dataset['xmin']/2
dataset['centerY'] = dataset['ymax']+dataset['ymin']/2
dataset = dataset[['name', 'centerX','centerY']]
#save new_dataset in a csv file
dataset.to_csv('new_dataset_utils\\positions.csv', index=False)