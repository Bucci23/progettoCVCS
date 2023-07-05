import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
import pandas as pd
import clustering_utils as utils
#read image data from csv file
def get_x_and_y():
    dataset = pd.read_csv('spectral_clustering.csv')
    #print(dataset['name'])
    annotation_dict = {}
    grouped_by_name = dataset.groupby('name')
    for name in grouped_by_name.groups.keys():
        annotation_dict[name] = grouped_by_name.get_group(name).drop('name', axis=1)
    print(len(annotation_dict))
    classes = pd.read_csv('classes.csv')
    print(len(classes))
    length = 0
    for image_index in range(len(annotation_dict)):
        image_name = list(annotation_dict.keys())[image_index]
        image_data = annotation_dict[image_name]
        X = image_data['centerY'].to_numpy().reshape(1,-1)
        if(length < X.shape[1]):
            length = X.shape[1]
    print(length)
    X_dataset = np.zeros((len(annotation_dict), length))
    # print(X_dataset.shape)
    for image_index in range(len(annotation_dict)):
        image_name = list(annotation_dict.keys())[image_index]
        image_data = annotation_dict[image_name]
        X = image_data['centerY'].to_numpy().reshape(1,-1)
        X_dataset[image_index, 0:X.shape[1]] = X
    # print(X_dataset)
    y_dataset = classes['shelves'].to_numpy()
    # print(y_dataset.shape)
    return X_dataset, y_dataset

if __name__ == '__main__':
    X_dataset, y_dataset = get_x_and_y()
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50)
    pcaed_X = pca.fit_transform(X_dataset)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(pcaed_X, y_dataset, test_size=0.2)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    clf = RandomForestClassifier()#Posso omettere i parametri
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print('forest:',accuracy_score(y_test, y_pred))     #accuracy
    #confusion_matrix(y_test, y_pred)    #confusion matrix
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error
    clf = LinearRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = mean_absolute_error(y_test, y_pred)
    print('MAE:',score)
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import cross_val_score
    
    clf = LogisticRegression(max_iter=10000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print('LOGREG',accuracy_score(y_test, y_pred)) 
