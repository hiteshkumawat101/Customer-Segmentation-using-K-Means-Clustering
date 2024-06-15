

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
cust_data = pd.read_csv('path_to_your_file')

# Display the first few rows of the dataset
print(cust_data.head())

# Check the shape of the dataset
print(cust_data.shape)

# Generate descriptive statistics
print(cust_data.describe())

# Check for missing values
print(cust_data.isna().sum())

# Encode 'Gender' column: Male -> 0, Female -> 1
cust_data.replace({'Gender': {'Male': 0, 'Female': 1}}, inplace=True)
print(cust_data.head())

# Extract features for clustering
X = cust_data.iloc[:, [3, 4]].values
print(X)

# List to hold the within-cluster sum of squares (WCSS)
wcss = []

# Calculate WCSS for different number of clusters
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=30)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Print WCSS values
print(wcss)

# Plot the elbow method graph
sns.set()
plt.plot(range(1, 20), wcss)
plt.xlabel('Number of Clusters')
plt.xticks(range(1, 21))
plt.xlim(1, 20)
plt.show()

# Apply KMeans clustering with the optimal number of clusters (5) as we see this point have a sharp edge so this will be optimal .
model = KMeans(n_clusters=5, init='k-means++', random_state=30)
y = model.fit_predict(X)
print(y)

# Visualize the clusters
plt.figure(figsize=(10, 10))
plt.scatter(X[y == 0, 0], X[y == 0, 1], s=50, c='green', label='Cluster 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[y == 2, 0], X[y == 2, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[y == 3, 0], X[y == 3, 1], s=50, c='yellow', label='Cluster 3')
plt.scatter(X[y == 4, 0], X[y == 4, 1], s=50, c='cyan', label='Cluster 4')
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='black', marker='x', label='Cluster Centers')
plt.legend()
plt.show()
