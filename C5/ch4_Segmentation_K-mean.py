import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load initial file
datamart_rfm = pd.read_csv('rfm_datamart.csv')
print(datamart_rfm.head())
print(datamart_rfm.info())
print(datamart_rfm.describe())

'''Key k-means assumptions: 
- Symmetric distribution of variables (not skewed)
- Variables with same average values
- Variables with same variance'''

# 0. Detect skewed variables on one chart
plt.subplot(3, 1, 1)
sns.distplot(datamart_rfm['Recency'])
plt.subplot(3, 1, 2)
sns.distplot(datamart_rfm['Frequency'])
plt.subplot(3, 1, 3)
sns.distplot(datamart_rfm['MonetaryValue'])
plt.show()

''' 1. Data pre-processing for clustering 
1. Unskew the data - log transformation
2. Standardize to the same average values
3. Scale to the same standard deviation
4. Store as a separate array to be used for clustering
'''

# 1. Unskew the data - log transformation
datamart_log = np.log(datamart_rfm)
# Centering variables with different means and Scaling variables with different variance using StandardScaler
# Initialize a standard scaler
scaler = StandardScaler()
# and fit it
scaler.fit(datamart_log)
# Scale and center the data
datamart_normalized = scaler.transform(datamart_log)
# Create a pandas DataFrame
datamart_normalized = pd.DataFrame(data=datamart_normalized,
                                   index=datamart_rfm.index,
                                   columns=datamart_rfm.columns)

# Visualize the normalized variables
# Plot recency distribution
plt.subplot(3, 1, 1)
sns.distplot(datamart_normalized['Recency'])
# Plot frequency distribution
plt.subplot(3, 1, 2)
sns.distplot(datamart_normalized['Frequency'])
# Plot monetary value distribution
plt.subplot(3, 1, 3)
sns.distplot(datamart_normalized['MonetaryValue'])
# Show the plot
plt.show()

print('mean: ', datamart_normalized.mean(axis=0).round(2))
print('std: ', datamart_normalized.std(axis=0).round(2))

''' Key steps for K-mean clustering after data pre-processing:
2. Choosing a number of clusters
3. Running k-means clustering on pre-processed data
4. Analyzing average RFM values of each cluster
'''

''' Methods to define the number of clusters
- Visual methods - elbow criterion
- Mathematical methods - silhouette coefficient
- Experimentation and interpretation
'''

# Running k-means with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=1)
# Compute k-means clustering on pre-processed data
kmeans.fit(datamart_normalized)
# Extract cluster labels from labels_ attribute
cluster_labels = kmeans.labels_
# Create a cluster label column in the original DataFrame:
datamart_rfm_k2 = datamart_rfm.assign(Cluster=cluster_labels)
# Calculate average RFM values and size for each cluster:
print(datamart_rfm_k2.groupby(['Cluster']).agg({'Recency': 'mean',
                                          'Frequency': 'mean',
                                          'MonetaryValue': ['mean', 'count'],
                                          }).round(0))

# 2. Choosing number of clusters by elbow criterion
''' 1st technique - Elbow criterion method
- Plot the number of clusters against within-cluster sum-of-squared-errors (SSE) - sum of squared distances from every data point to their cluster center
- Identify an "elbow" in the plot (Elbow - a point representing an "optimal" number of clusters)'''
# Fit KMeans and calculate SSE for each *k*
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1)
    kmeans.fit(datamart_normalized)
    sse[k] = kmeans.inertia_ # sum of squared distances to closest cluster center
# Plot SSE for each *k*
plt.title('The Elbow Method')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

# Running k-means with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=1)
# Compute k-means clustering on pre-processed data
kmeans.fit(datamart_normalized)
# Extract cluster labels from labels_ attribute
cluster_labels = kmeans.labels_
# Create a cluster label column in the original DataFrame:
datamart_rfm_k3 = datamart_rfm.assign(Cluster=cluster_labels)
# Calculate average RFM values and size for each cluster:
print(datamart_rfm_k3.groupby(['Cluster']).agg({'Recency': 'mean',
                                          'Frequency': 'mean',
                                          'MonetaryValue': ['mean', 'count'],
                                          }).round(0))

# Snake plot to understand and compare segments
''' 2nd technique - Snake plot
- Market research technique to compare different segments
- Visual representation of each segment's attributes
- Need to first normalize data (center & scale)
- Plot each cluster's average normalized values of each attribute
'''
# Transform datamart_normalized as DataFrame and add a Cluster column
datamart_normalized = pd.DataFrame(datamart_normalized,
                                   index=datamart_rfm.index,
                                   columns=datamart_rfm.columns)
datamart_normalized['Cluster'] = datamart_rfm_k3['Cluster']

# Melt the data into a long format so RFM values and metric names are stored in 1 column each
datamart_melt = pd.melt(datamart_normalized.reset_index(),
                        id_vars=['CustomerID', 'Cluster'],
                        value_vars=['Recency', 'Frequency', 'MonetaryValue'],
                        var_name='Attribute',
                        value_name='Value')
# Visualize a snake plot
plt.title('Snake plot of standardized variables')
sns.lineplot(x="Attribute", y="Value", hue='Cluster', data=datamart_melt)
plt.show()

# Calculate relative importance of each attribute
'''Useful technique to identify relative importance of each segment's attribute
- Calculate average values of each cluster
- Calculate average values of population
- Calculate importance score by dividing them and subtracting 1 (ensures 0 is returned when cluster average equals population average)
'''
cluster_avg = datamart_rfm_k3.groupby(['Cluster']).mean()
population_avg = datamart_rfm.mean()
relative_imp = cluster_avg / population_avg - 1
print(relative_imp.round(2))
# Plot a heatmap for easier interpretation
plt.figure(figsize=(8, 2))
plt.title('Relative importance of attributes')
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()


'''Another Project'''
# Implement end-to-end segmentation solution
# Updated RFM data in datamart_rfmt.csv -> Same RFM values plus additional Tenure variable
# Tenure - time since the first transaction (Defines how long the customer has been with the company)



# Load initial file
datamart_rfmt = pd.read_csv('datamart_rfmt.csv')
# Pre-process data
# Import StandardScaler
from sklearn.preprocessing import StandardScaler
# Apply log transformation
datamart_rfmt_log = np.log(datamart_rfmt)
# Initialize StandardScaler and fit it
scaler = StandardScaler(); scaler.fit(datamart_rfmt_log)
# Transform and store the scaled data as datamart_rfmt_normalized
datamart_rfmt_normalized = scaler.transform(datamart_rfmt_log)

# Calculate and plot sum of squared errors
# Fit KMeans and calculate SSE for each k between 1 and 10
for k in range(1, 11):
    # Initialize KMeans with k clusters and fit it
    kmeans = KMeans(n_clusters=k, random_state=1).fit(datamart_rfmt_normalized)
    # Assign sum of squared distances to k element of the sse dictionary
    sse[k] = kmeans.inertia_
# Add the plot title, x and y axis labels
plt.title('The Elbow Method')
plt.xlabel('k')
plt.ylabel('SSE')
# Plot SSE values for each k stored as keys in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

# Build 4-cluster solution
# Import KMeans
from sklearn.cluster import KMeans
# Initialize KMeans
kmeans = KMeans(n_clusters=4, random_state=1)
# Fit k-means clustering on the normalized data set
kmeans.fit(datamart_rfmt_normalized)
# Extract cluster labels
cluster_labels = kmeans.labels_

# Analyze the segments
# Create a new DataFrame by adding a cluster label column to datamart_rfmt
datamart_rfmt_k4 = datamart_rfmt.assign(Cluster=cluster_labels)
# Group by cluster
grouped = datamart_rfmt_k4.groupby(['Cluster'])
# Calculate average RFMT values and segment sizes for each cluster
print(grouped.agg({'Recency': 'mean',
                   'Frequency': 'mean',
                   'MonetaryValue': 'mean',
                   'Tenure': ['mean', 'count']}).round(1))