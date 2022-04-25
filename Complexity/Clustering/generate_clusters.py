import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Reading IRIS data
iris = pd.read_csv('../data/iris.csv')
iris_x = iris.to_numpy()
# Extracting features
iris_data = iris.iloc[:, [0, 1, 2, 3]].values
# Extracting labels
iris_target_names = iris.iloc[:, 4].values

class_names = ['setosa', 'versicolor', 'virginica']

# Select a class to cluster
for id in range(len(class_names)):
    # Select which class to cluster
    all_indexes = pd.DataFrame([x for x in range(len(iris_data))])
    current_indexes = all_indexes[iris.iloc[:, 4] == class_names[id]].iloc[:, 0].values
    other_indexes = all_indexes[iris.iloc[:, 4] != class_names[id]].iloc[:, 0].values
    x = iris_data[current_indexes, :]

    # Select a number of clusters and compute using KMeans
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    # Feed a subset of the data to KMeans
    y_kmeans = kmeans.fit_predict(x)

    indexed_y_kmeans = pd.Series(y_kmeans, index=current_indexes)
    all_filled_indexed_y_kmeans = indexed_y_kmeans.reindex(pd.Series(range(len(iris)))).fillna(value=-1).astype('int64')
    print()

    for i in range(n_clusters):
        current_cluster = iris[all_filled_indexed_y_kmeans == i]
        other_class_data = iris[all_filled_indexed_y_kmeans == -1]
        print('Saving data to:')
        print(f'class_{class_names[id]}_cluster_{i}.csv')
        cluster_vs_other_data = pd.concat([current_cluster, other_class_data])
        print(cluster_vs_other_data)
        cluster_vs_other_data.reset_index().to_csv(f'../data/class_{class_names[id]}_cluster_{i}.csv')
