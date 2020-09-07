import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load initial file
telco_raw = pd.read_csv('telco.csv')
print(telco_raw.head())
print(telco_raw.info())
print(telco_raw.describe())
print(telco_raw.dtypes)

'''0. Pre-processing'''

# convert string to numerical for numerical colunms
telco_raw['tenure'] = pd.to_numeric(telco_raw['tenure'], errors='coerce')
telco_raw['MonthlyCharges'] = pd.to_numeric(telco_raw['MonthlyCharges'], errors='coerce')
telco_raw['TotalCharges'] = pd.to_numeric(telco_raw['TotalCharges'], errors='coerce')

# Store customerID and Churn column names
custid = ['customerID']
target = ['Churn']
# Replace 'no' with 0 and 'yes' with 1 in 'Churn'
telco_raw['Churn'] = telco_raw['Churn'].replace({"No": 0, "Yes": 1})

# Separate categorical and numeric column names as lists.
# Assign to categorical the column names that have less than 5 unique values.
categorical = telco_raw.nunique()[telco_raw.nunique() < 5].keys().tolist()
# Remove target from the list of categorical variables
categorical.remove(target[0])
# Store numerical column names. Assign to numerical all column names that are not in the custid, target and categorical.
numerical = [col for col in telco_raw.columns if col not in custid+target+categorical]

# Perform one-hot encoding to categorical variables
telco_raw = pd.get_dummies(data=telco_raw, columns=categorical, drop_first=True)
# Initialize StandardScaler instance
scaler = StandardScaler()
# Fit and transform the scaler on numerical columns
scaled_numerical = scaler.fit_transform(telco_raw[numerical])
# Build a DataFrame from scaled_numerical
scaled_numerical = pd.DataFrame(scaled_numerical, columns=numerical)
print(scaled_numerical)

# Drop non-scaled numerical columns
telco_raw = telco_raw.drop(columns=numerical, axis=1)
# Merge the non-numerical with the scaled numerical data
telco = telco_raw.merge(right=scaled_numerical, how='left', left_index=True, right_index=True)

# To avoid "ValueError: Input contains NaN, infinity or a value too large for dtype('float32')"
# I decided to check Nan values
print(telco.isnull().sum()) # 11 in TotalCharges
print(telco.shape[0]) # 7043
telco = telco.dropna()
print(telco.isnull().sum()) # 0 in TotalCharges
print(telco.shape[0]) # 7032
telco = telco.drop('customerID', axis=1)


'''
Supervised learning steps
1. Split data to training and testing
2. Initialize the model
3. Fit the model on the training data
4. Predict values on the testing data
5. Measure model performance on testing data
'''

from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Create a feature variable X which holds all of the features of telco by dropping the target variable 'Churn' from
# telco.
X = telco.drop('Churn', axis=1)
# Create a target variable y which holds the values of the target variable - 'Churn'.
Y = telco['Churn']

# 1. Split data to training and testing
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25)

# Ensure training dataset has only 75% of original X data
print(train_X.shape[0] / X.shape[0])
# Ensure testing dataset has only 25% of original X data
print(test_X.shape[0] / X.shape[0])

# 2. Initialize the model
mytree = tree.DecisionTreeClassifier()
# 3. Fit the model on the training data
treemodel = mytree.fit(train_X, train_Y)
# 4. Predict values on the testing data
pred_Y = treemodel.predict(test_X)
# 5. Measure model performance on testing data
print(accuracy_score(test_Y, pred_Y))

'''Unsupervised learning steps
1. Initialize the model
2. Fit the model
3. Assign cluster values
4. Explore results '''

from sklearn.cluster import KMeans
# 1. Initialize the model and # 2. Fit the model
kmeans = KMeans(n_clusters=2, random_state=0).fit(telco)


