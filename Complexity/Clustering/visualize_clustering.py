import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reading IRIS data
iris = pd.read_csv('../data/iris.csv')
# Extracting features
iris_data = iris.iloc[:, [0, 1, 2, 3]].values
# Extracting labels
iris_target_names = iris.iloc[:, 4].values

# Select a class to cluster
class_names = ['setosa', 'versicolor', 'virginica']

# Select a class to cluster
for id in range(len(class_names)):
    # Select which class to cluster
    x = iris_data[iris.iloc[:, 4] == class_names[id], :]
    other_class = iris_data[iris.iloc[:, 4] != class_names[id], :]

    # The indices of the features that we are plotting
    x_index = 0
    y_index = 1
    z_index = 2

    # Select a number of clusters and compute using KMeans
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Feed a subset of the data to KMeans
    y_kmeans = kmeans.fit_predict(x[:, [x_index, y_index, z_index]])

    # Set up a figure
    fig = plt.figure(figsize=plt.figaspect(0.5))
    x_lim = [4, 9]
    y_lim = [1, 6]
    z_lim = [0, 8]

    # First plot to show unlabelled data
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    ax.scatter3D(x[:, x_index], x[:, y_index], x[:, z_index], label='Current class')
    ax.scatter3D(other_class[:, x_index], other_class[:, y_index], other_class[:, z_index], color='red', label='Other classes')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)
    plt.legend()

    # Second plot to show clusters
    ax = fig.add_subplot(2, 3, 2, projection='3d')
    # Visualising the clusters
    for i in range(n_clusters):
        ax.scatter3D(x[y_kmeans == i, x_index], x[y_kmeans == i, y_index], x[y_kmeans == i, z_index], label=f'Cluster {i+1}')
    ax.scatter3D(other_class[:, x_index], other_class[:, y_index], other_class[:, z_index], color='red', label='Other classes')

    # Plotting the centroids of the clusters
    ax.scatter3D(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], c='black', label='Centroids')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_zlim(z_lim)

    plt.legend()

    # Second row of plots for to show separate clusters
    for i in range(n_clusters):
        ax = fig.add_subplot(2, 3, i+4, projection='3d')
        # Visualising the clusters
        ax.scatter3D(x[y_kmeans == i, x_index], x[y_kmeans == i, y_index], x[y_kmeans == i, z_index], label=f'Cluster {i+1}')
        ax.scatter3D(other_class[:, x_index], other_class[:, y_index], other_class[:, z_index], color='red', label='Other classes')

        # Plotting the centroids of the clusters
        ax.scatter3D(kmeans.cluster_centers_[i, 0], kmeans.cluster_centers_[i, 1], kmeans.cluster_centers_[i, 2], c='black', label='Centroids')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)

        plt.legend()

        if id == 0:
            print(f'Setosa: Cluster {i+1}')
        elif id == 1:
            print(f'Versicolor: Cluster {i+1}')
        else:
            print(f'Virginica: Cluster {i+1}')
        
        clustered_x = x[y_kmeans == i, :]
        print(clustered_x[:, [x_index, y_index, z_index]])

    plt.show()
