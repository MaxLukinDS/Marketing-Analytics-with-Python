import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF

# Load initial file
wholesale = pd.read_csv('wholesale.csv')
print(wholesale.head())
print(wholesale.info())

wholesale = wholesale.drop(wholesale[['Channel', 'Region']], axis=1)

# Explore variables
print(wholesale.agg(['mean', 'std']).round(0))

# Get the statistics
averages = wholesale.mean()
std_devs = wholesale.std()
x_names = wholesale.columns
x_ix = np.arange(wholesale.shape[1])

# Create column names list and same length integer list
x_names = wholesale.columns
x_ix = np.arange(wholesale.shape[1])
# Plot the averages data in gray and standard deviations in orange
plt.bar(x=x_ix-0.2, height=averages, color='grey', label='Average', width=0.4)
plt.bar(x=x_ix+0.2, height=std_devs, color='orange', label='Standard Deviation', width=0.4)
# Add x-axis labels and rotate
plt.xticks(ticks=x_ix, labels=x_names, rotation=90)
# Add the legend and display the chart
plt.legend()
plt.show()

# Plot the pairwise relationships between the variables
sns.pairplot(wholesale, diag_kind='kde')
# Display the chart
plt.show()

'''Model assumptions:
First we'll start with K-means
K-means clustering works well when data is 
1) ~normally distributed (no skew), and 
2) standardized (mean = 0, standard deviation = 1)
'''
# Unskewing data with log-transformation
# First option - log transformation
wholesale_log = np.log(wholesale)
sns.pairplot(wholesale_log, diag_kind='kde')
plt.show()

# Unskewing data with Box-Cox transformation
# Second option - Box-Cox transformation
def boxcox_df(x):
    x_boxcox, _ = stats.boxcox(x)
    return x_boxcox
wholesale_boxcox = wholesale.apply(boxcox_df, axis=0)
sns.pairplot(wholesale_boxcox, diag_kind='kde')
plt.show()

# Scale the data
scaler = StandardScaler()
# Fit the initialized `scaler` instance on the Box-Cox transformed dataset
scaler.fit(wholesale_boxcox)
# Transform and store the scaled dataset as `wholesale_scaled`
wholesale_scaled = scaler.transform(wholesale_boxcox)
# Create a `pandas` DataFrame from the scaled dataset
wholesale_scaled_df = pd.DataFrame(data=wholesale_scaled,
                                   index=wholesale_boxcox.index,
                                   columns=wholesale_boxcox.columns)
# Print the mean and standard deviation for all columns
print(wholesale_scaled_df.agg(['mean', 'std']).round())


'''Segmentation steps with K-means'''
# Determine the optimal number of clusters
# Create empty sse dictionary
sse = {}
# Fit KMeans algorithm on k values between 1 and 11
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=333)
    kmeans.fit(wholesale_scaled_df)
    sse[k] = kmeans.inertia_
# Add the title to the plot
plt.title('Elbow criterion method chart')
# Create and display a scatter plot
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()

# Build segmentation with k-means clustering
'''Unsupervised learning steps
1. Initialize the model
2. Fit the model
3. Assign cluster values
4. Explore results'''
# Initialize `KMeans` with 4 clusters
kmeans=KMeans(n_clusters=4, random_state=123)
# Fit the model on the pre-processed dataset
kmeans.fit(wholesale_scaled_df)
# Assign the generated labels to a new column
wholesale_kmeans4 = wholesale.assign(segment=kmeans.labels_)

# Analyze average K-means segmentation attributes
kmeans4_averages = wholesale_kmeans4.groupby(['segment']).mean().round(0)
# Print the average column values per each segment
print(kmeans4_averages)
# Create a heatmap on the average column values per each segment
ax = plt.axes()
sns.heatmap(kmeans4_averages.T, cmap='YlGnBu', ax=ax)
ax.set_title('heatmap for k-means')
plt.show()

'''Second model - Non-negative matrix factorization (NMF) - can be used on raw data, especially if the matrix is sparse'''
'''Segmentation steps with NMF'''
# Initialize NMF instance with 4 components
nmf = NMF(4)
# Fit the model on the wholesale sales data
nmf.fit(wholesale)
# Extract the components
components = pd.DataFrame(data=nmf.components_, columns=wholesale.columns)

# Extracting segment assignment:
# Create the W matrix
W = pd.DataFrame(data=nmf.transform(wholesale), columns=components.index)
W.index = wholesale.index
# Assign the column name where the corresponding value is the largest
wholesale_nmf4 = wholesale.assign(segment=W.idxmax(axis=1))
# Calculate the average column values per each segment
nmf4_averages = wholesale_nmf4.groupby('segment').mean().round(0)
# Plot the average values as heatmap
ax = plt.axes()
sns.heatmap(nmf4_averages.T, cmap='YlGnBu', ax=ax)
ax.set_title('heatmap for NMF')
plt.show()






