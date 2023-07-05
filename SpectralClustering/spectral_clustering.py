import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import pairwise_distances
from sklearn.datasets import make_blobs
import pandas as pd
import clustering_utils as utils
#read image data from csv file
dataset = pd.read_csv('spectral_clustering.csv')
#print(dataset['name'])
annotation_dict = {}
grouped_by_name = dataset.groupby('name')
for name in grouped_by_name.groups.keys():
    annotation_dict[name] = grouped_by_name.get_group(name).drop('name', axis=1)
#print(annotation_dict)
for image_index in range(len(annotation_dict)):
    image_name = list(annotation_dict.keys())[image_index]
    image_data = annotation_dict[image_name]
    X = image_data['centerY'].to_numpy()
    # print(X)
    # Compute pairwise distances
    distances = pairwise_distances(X.reshape(-1, 1))
    n=len(X)
    print(n/7)
    # Perform spectral clustering with eigengap method for number of clusters
    affinity_matrix = utils.getAffinityMatrix(X.reshape(-1, 1), k=int(n/6))
    # eigenvalues, eigenvectors = np.linalg.eig(affinity_matrix)
    # sorted_indices = np.argsort(eigenvalues)[::-1]  # Sort eigenvalues in descending order
    # eigen_gaps = np.diff(np.sort(eigenvalues)[::-1])[0:10]
    # k = np.argmax(eigen_gaps) + 1  # Number of clusters
    k, _, _ = utils.eigenDecomposition(affinity_matrix, plot=False, topK=5)
    print(k)
    spectral_model = SpectralClustering(n_clusters=k[0], affinity='precomputed', random_state=0)
    labels = spectral_model.fit_predict(affinity_matrix)
    # print(labels)
    # Plot results: load the image and plot the points colored according to their cluster
    image = plt.imread(f'images\\{image_name}')
    plt.title(image_index)
    plt.imshow(image)
    plt.scatter(image_data['centerX'], image_data['centerY'], c=labels, s=20, cmap='plasma')
    plt.show()