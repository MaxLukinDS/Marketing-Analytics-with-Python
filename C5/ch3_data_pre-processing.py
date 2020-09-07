import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load initial file
datamart_rfm = pd.read_csv('rfm_datamart.csv')
print(datamart_rfm.head())
print(datamart_rfm.info())
print(datamart_rfm.describe())

'''Key k-means assumptions: 
- Symmetric distribution of variables (not skewed)
- Variables with same average values
- Variables with same variance'''

''' Data pre-processing for clustering 
1. Unskew the data - log transformation
2. Standardize to the same average values
3. Scale to the same standard deviation
4. Store as a separate array to be used for clustering
'''

# Exploring distribution of Recency
sns.distplot(datamart_rfm['Recency'])
plt.show()
# Exploring distribution of Frequency
sns.distplot(datamart_rfm['Frequency'])
plt.show()
# Exploring distribution of Frequency
sns.distplot(datamart_rfm['MonetaryValue'])
plt.show()

# Detect skewed variables on one chart
plt.subplot(3, 1, 1)
sns.distplot(datamart_rfm['Recency'])
plt.subplot(3, 1, 2)
sns.distplot(datamart_rfm['Frequency'])
plt.subplot(3, 1, 3)
sns.distplot(datamart_rfm['MonetaryValue'])
plt.show()

# Data transformations to manage skewness -> Logarithmic transformation (positive values only)
recency_log = np.log(datamart_rfm['Frequency'])
frequency_log = np.log(datamart_rfm['Frequency'])
monetaryvalue_log = np.log(datamart_rfm['Frequency'])

plt.subplot(3, 1, 1)
sns.distplot(recency_log)
plt.subplot(3, 1, 2)
sns.distplot(frequency_log)
plt.subplot(3, 1, 3)
sns.distplot(monetaryvalue_log)
plt.show()

# Centering variables with different means and Scaling variables with different variance using StandardScaler
scaler = StandardScaler()
scaler.fit(datamart_rfm)
datamart_normalized = scaler.transform(datamart_rfm)
print('mean: ', datamart_normalized.mean(axis=0).round(2))
print('std: ', datamart_normalized.std(axis=0).round(2))


