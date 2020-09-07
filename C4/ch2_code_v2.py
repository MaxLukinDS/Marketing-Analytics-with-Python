import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

# Load initial file
telco_raw = pd.read_csv('telco.csv')
print(telco_raw.columns)

# telco_raw.drop(['InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No internet service', 'OnlineBackup_No internet service', 'DeviceProtection_No internet service', 'TechSupport_No internet service', 'StreamingTV_No internet service', 'StreamingMovies_No internet service'], axis=1, inplace=True)

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
telcom = telco.dropna()
print(telcom.columns)

# Print the unique Churn values
print(set(telcom['Churn']))
# Calculate the ratio size of each churn group
telcom.groupby(['Churn']).size() / telcom.shape[0] * 100
# Import the function for splitting data to train and test

# Split the data into train and test
train, test = train_test_split(telcom, test_size=.25)

# Store column names from `telcom` excluding target variable and customer ID
cols = [col for col in telcom.columns if col not in custid + target]
# Extract training features
train_X = train[cols]
# Extract training target
train_Y = train[target]
# Extract testing features
test_X = test[cols]
# Extract testing target
test_Y = test[target]

'''Modeling steps for Logistic Regression
1. Split data to training and testing
2. Initialize the model
3. Fit the model on the training data
4. Predict values on the testing data
5. Measure model performance on testing data'''

# Initialize Logistic Regression instance
logreg = LogisticRegression()
# Fit the model on the training data
logreg.fit(train_X, train_Y)

'''Key Model performance metrics:
Accuracy - The % of correctly predicted labels (both Churn and non Churn)
Precision - The % of total model's positive class predictions (here - predicted as Churn) that were
correctly classified
Recall - The % of total positive class samples (all churned customers) that were correctly classified'''

# Measuring model accuracy
pred_train_Y = logreg.predict(train_X)
pred_test_Y = logreg.predict(test_X)
train_accuracy = accuracy_score(train_Y, pred_train_Y)
test_accuracy = accuracy_score(test_Y, pred_test_Y)
print('Training accuracy:', round(train_accuracy, 4))
print('Test accuracy:', round(test_accuracy, 4))

# Measuring precision and recall
train_precision = round(precision_score(train_Y, pred_train_Y), 4)
test_precision = round(precision_score(test_Y, pred_test_Y), 4)
train_recall = round(recall_score(train_Y, pred_train_Y), 4)
test_recall = round(recall_score(test_Y, pred_test_Y), 4)
print('Training precision: {}, Training recall: {}'.format(train_precision, train_recall))
print('Test precision: {}, Test recall: {}'.format(train_recall, test_recall))

'''Regularization
- Introduces penalty coefficient in the model building phase 
- Addresses over-fitting (when patterns are "memorized by the model and not predicting the results")
- Some regularization techniques also perform feature selection e.g. L1
- Makes the model more generalizable to unseen samples
'''

'''L1 regularization or also called LASSO can be called explicitly, and this approach performs feature
selection by shrinking some of the model coefficients to zero.'''
logreg = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
logreg.fit(train_X, train_Y)

# C parameter needs to be tuned to find the optimal value
'''Tuning L1 regularization'''
C = [1, .5, .25, .1, .05, .025, .01, .005, .0025]
l1_metrics = np.zeros((len(C), 5))
l1_metrics[:, 0] = C
for index in range(0, len(C)):
    logreg = LogisticRegression(penalty='l1', C=C[index], solver='liblinear')
    logreg.fit(train_X, train_Y)
    pred_test_Y = logreg.predict(test_X)
    l1_metrics[index, 1] = np.count_nonzero(logreg.coef_)
    l1_metrics[index, 2] = accuracy_score(test_Y, pred_test_Y)
    l1_metrics[index, 3] = precision_score(test_Y, pred_test_Y)
    l1_metrics[index, 4] = recall_score(test_Y, pred_test_Y)

col_names = ['C', 'Non-Zero Coeffs', 'Accuracy', 'Precision', 'Recall']
print(pd.DataFrame(l1_metrics, columns=col_names))


'''Modeling steps for Decision Tree Model 
1. Split data to training and testing
2. Initialize the model
3. Fit the model on the training data
4. Predict values on the testing data
5. Measure model performance on testing data'''

# Initialize the Decision Tree model
mytree = DecisionTreeClassifier()
# Fit the model on the training data
treemodel = mytree.fit(train_X, train_Y)

# Measuring model accuracy
pred_train_Y = mytree.predict(train_X)
pred_test_Y = mytree.predict(test_X)
train_accuracy = accuracy_score(train_Y, pred_train_Y)
test_accuracy = accuracy_score(test_Y, pred_test_Y)
print('Training accuracy:', round(train_accuracy, 4))
print('Test accuracy:', round(test_accuracy, 4))

'''Conclusion: This indicates that the Tree memorizes the paterns and  rules for the training data perfectly, 
but failed to generalized the rules for the testing data.'''

# Measuring precision and recall
train_precision = round(precision_score(train_Y, pred_train_Y), 4)
test_precision = round(precision_score(test_Y, pred_test_Y), 4)
train_recall = round(recall_score(train_Y, pred_train_Y), 4)
test_recall = round(recall_score(test_Y, pred_test_Y), 4)
print('Training precision: {}, Training recall: {}'.format(train_precision, train_recall))
print('Test precision: {}, Test recall: {}'.format(train_recall, test_recall))

'''Conclusion: TRecall means the number of total churn instances correctly captured by the model. 
We can see that the model is very presice with its prediction, but fails to identify 
a bit less than a half of the churn customers.'''

'''Decision tree is very prone to overfitting as it will build rules 
that will memorize all the patterns down to each observation level. -> 
Tree depth parameter tuning:'''
depth_list = list(range(2, 15))
depth_tuning = np.zeros((len(depth_list), 4))
depth_tuning[:, 0] = depth_list
for index in range(len(depth_list)):
    mytree = DecisionTreeClassifier(max_depth=depth_list[index])
    mytree.fit(train_X, train_Y)
    pred_test_Y = mytree.predict(test_X)
    depth_tuning[index, 1] = accuracy_score(test_Y, pred_test_Y)
    depth_tuning[index, 2] = precision_score(test_Y, pred_test_Y)
    depth_tuning[index, 3] = recall_score(test_Y, pred_test_Y)

col_names = ['Max_Depth', 'Accuracy', 'Precision', 'Recall']
print(pd.DataFrame(depth_tuning, columns=col_names))

# Plotting decision tree rules
exported = tree.export_graphviz(decision_tree=mytree,
                                out_file=None,
                                feature_names=cols,
                                precision=1,
                                class_names=['Not churn', 'Churn'],
                                filled=True)
graph = graphviz.Source(exported)
display(graph)

''' Logistic regression coefficients - Logistic regression returns beta coefficients'''

'''Extracting logistic regression coefficients'''
# Coefficients can be extracted using .coef_ method on fitted Logistic Regression instance
print(logreg.coef_)

'''Transforming logistic regression coefficients'''
# Combine feature names and coefficients into pandas DataFrame
feature_names = pd.DataFrame(train_X.columns, columns=['Feature'])
log_coef = pd.DataFrame(np.transpose(logreg.coef_), columns=['Coefficient'])
coefficients = pd.concat([feature_names, log_coef], axis=1)
# Calculate exponent of the logistic regression coefficients
coefficients['Exp_Coefficient'] = np.exp(coefficients['Coefficient'])
# Remove coefficients that are equal to zero
coefficients = coefficients[coefficients['Coefficient'] != 0]
# Print the values sorted by the exponent coefficient
print(coefficients.sort_values(by=['Exp_Coefficient']))


