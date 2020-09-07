

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

# Load the teslo file
telco = pd.read_csv('Churn.csv')
telco.info()
telco.head()

# Group telco by 'Churn' and compute the mean
print(telco.groupby(['Churn']).mean())

# Adapt your code to compute the standard deviation
print(telco.groupby(['Churn']).std())

# Count the number of churners and non-churners by State
print(telco.groupby('State')['Churn'].value_counts())

# Visualize the distribution of 'Day_Mins'
sns.distplot(telco['Day_Mins'])
# Display the plot
plt.show()

# Visualize the distribution of 'Eve_Mins'
sns.distplot(telco['Eve_Mins'])
# Display the plot
plt.show()

# Visualize the distribution of 'Night_Mins'
sns.distplot(telco['Night_Mins'])
# Display the plot
plt.show()

# Visualize the distribution of 'Intl_Mins'
sns.distplot(telco['Intl_Mins'])
# Display the plot
plt.show()

# Create the box plot
sns.boxplot(x='Churn',
            y='CustServ_Calls',
            data=telco)
# Display the plot
plt.show()

# Create the box plot
sns.boxplot(x='Churn',
            y='CustServ_Calls',
            data=telco,
            sym="")
# Display the plot
plt.show()

# Add "Vmail_Plan" as a third variable
sns.boxplot(x='Churn',
            y='CustServ_Calls',
            data=telco,
            sym="",
            hue='Vmail_Plan')
# Display the plot
plt.show()

# Add "Intl_Plan" as a third variable
sns.boxplot(x='Churn',
            y='CustServ_Calls',
            data=telco,
            sym="",
            hue='Intl_Plan')
# Display the plot
plt.show()

# Replace 'no' with 0 and 'yes' with 1 in 'Vmail_Plan'
telco['Vmail_Plan'] = telco['Vmail_Plan'].replace({"no": 0, "yes": 1})
# Replace 'no' with 0 and 'yes' with 1 in 'Churn'
telco['Churn'] = telco['Churn'].replace({"no": 0, "yes": 1})
# Replace 'no' with 0 and 'yes' with 1 in 'Intl_Plan'
telco['Intl_Plan'] = telco['Intl_Plan'].replace({"no": 0, "yes": 1})

# Print the results to verify
print(telco['Vmail_Plan'].head())
print(telco['Churn'].head())
print(telco['Intl_Plan'].head())

# Perform one hot encoding on 'State'
telco_state = pd.get_dummies(telco['State'])
# Print the head of telco_state
print(telco_state)

# Drop the unnecessary features
telco = telco.drop(telco[['Area_Code', 'Phone', 'State']], axis=1)
# Verify dropped features
print(telco.columns)

# Create the new feature
telco['Avg_Night_Calls'] = telco['Night_Mins'] / telco['Night_Calls']
# Print the first five rows of 'Avg_Night_Calls'
print(telco['Avg_Night_Calls'].head())

# Import train_test_split
# Create a feature variable X which holds all of the features of telco by dropping the target variable 'Churn' from
# telco.
X = telco.drop('Churn', axis=1)
# Create a target variable y which holds the values of the target variable - 'Churn'.
y = telco['Churn']
# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Import RandomForestClassifier
# Instantiate the classifier
model1 = RandomForestClassifier()
# Fit to the training data
model1.fit(X_train, y_train)
prediction_test1 = model1.predict(X_test)

# Compute metrics
accuracy1 = metrics.accuracy_score(y_test, prediction_test1)
precision1 = metrics.precision_score(y_test, prediction_test1)
recall1 = metrics.recall_score(y_test, prediction_test1)
f1_1 = metrics.f1_score(y_test, prediction_test1)
print(" Metrics for the model 1:")
print(f" Accuracy Score {accuracy1:.5f}")
print(f" Precision Score {precision1:.5f}")
print(f" Recall Score {recall1:.5f}")
print(f" F1-Score {f1_1:.5f}")
print(" Confusion Matrix:")
print(confusion_matrix(y_test, prediction_test1))

# Create training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Instantiate the classifier
model2 = RandomForestClassifier()
# Fit to the training data
model2.fit(X_train, y_train)
prediction_test2 = model2.predict(X_test)
# Compute metrics
accuracy2 = metrics.accuracy_score(y_test, prediction_test2)
precision2 = metrics.precision_score(y_test, prediction_test2)
recall2 = metrics.recall_score(y_test, prediction_test2)
f1_2 = metrics.f1_score(y_test, prediction_test2)
print(" Metrics for the model 2:")
print(f" Accuracy Score {accuracy2:.5f}")
print(f" Precision Score {precision2:.5f}")
print(f" Recall Score {recall2:.5f}")
print(f" F1-Score {f1_2:.5f}")
print(" Confusion Matrix:")
print(confusion_matrix(y_test, prediction_test2))

# Generate the probabilities
y_pred_prob = model2.predict_proba(X_test)[:, 1]
# Calculate the roc metrics
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot the ROC curve
plt.plot(fpr, tpr)
# Add labels and diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.plot([0, 1], [0, 1], "k--")
plt.show()

# Create the hyperparameter grid
param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
# Call GridSearchCV
grid_search = GridSearchCV(model2, param_grid)
# Fit the model
grid_search.fit(X, y)
# Use the .best_params_ attribute of grid_search to identify the best combination of parameters
print(grid_search.best_params_)

# Create the hyperparameter grid
param_dist = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Call RandomizedSearchCV
random_search = RandomizedSearchCV(model2, param_dist)
# Fit the model
random_search.fit(X, y)
# Print best parameters
print(random_search.best_params_)

# Calculate feature importances
importances = model2.feature_importances_
# Create plot
plt.barh(range(X.shape[1]), importances)
plt.show()
# Improving the plot
# Sort importances
sorted_index = np.argsort(importances)
# Create labels
labels = X.columns[sorted_index]
# Create plot
plt.barh(range(X.shape[1]), importances[sorted_index], tick_label=labels)
plt.show()