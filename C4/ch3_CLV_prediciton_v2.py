import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load initial file
online = pd.read_csv('online-new.csv')
print(online.head())
print(online.info())

# online['InvoiceDate'] = pd.to_datetime(online['InvoiceDate'], errors='coerce')
online['InvoiceDate'] = pd.to_datetime(online['InvoiceDate'])
# Assign acquisition month cohort
def get_month(x): return pd.datetime(x.year, x.month, 1)
online['InvoiceMonth'] = online['InvoiceDate'].apply(get_month)
# Add a new column TotalSum = Quantity x UnitPrice
online['TotalSum'] = online['Quantity'] * online['UnitPrice']

grouping = online.groupby('CustomerID')['InvoiceMonth']
online['CohortMonth'] = grouping.transform('min')
print(online.head())

# Extract integer values from data: Define function to extract year, month and day integer values.
def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    day = df[column].dt.day
    return year, month, day

# Assign time offset value
invoice_year, invoice_month, _ = get_date_int(online, 'InvoiceMonth')
cohort_year, cohort_month, _ = get_date_int(online, 'CohortMonth')
years_diff = invoice_year - cohort_year
months_diff = invoice_month - cohort_month

online['CohortIndex'] = years_diff * 12 + months_diff + 1
print(online.head())

# Count monthly active customers from each cohort
grouping = online.groupby(['CohortMonth', 'CohortIndex'])
cohort_data = grouping['CustomerID'].apply(pd.Series.nunique)
cohort_data = cohort_data.reset_index()
cohort_counts = cohort_data.pivot(index='CohortMonth',
                                  columns='CohortIndex',
                                  values='CustomerID')
print(cohort_counts)

'''Build retention and churn tables'''
# Extract cohort sizes from the first column of cohort_counts
cohort_sizes = cohort_counts.iloc[:, 0]
# Calculate retention table by dividing the counts with the cohort sizes
retention = cohort_counts.divide(cohort_sizes, axis=0)
# Calculate churn table
churn = 1 - retention
# Print the retention table
print(retention)

# Explore retention and churn
# Calculate the mean retention rate
retention_rate = retention.iloc[:, 1:].mean().mean()
# Calculate the mean churn rate
churn_rate = churn.iloc[:, 1:].mean().mean()
# Print rounded retention and churn rates
print('Retention rate: {:.2f}; Churn rate: {:.2f}'.format(retention_rate, churn_rate))

''' Basic CLV Calculation: CLV = Average Revenue * Average Customer Lifespan
In our case - we'll skip the profit margin for simplicity abd use revenue-based CLV formula '''
# Calculate monthly spend per customer
monthly_revenue = online.groupby(['CustomerID', 'InvoiceMonth'])['TotalSum'].sum().mean()
# Calculate average monthly spend
monthly_revenue = np.mean(monthly_revenue)
# Define lifespan to 36 months for simplicity
lifespan_months = 36
# Calculate basic CLV
clv_basic = monthly_revenue * lifespan_months
# Print basic CLV value
print('Average basic CLV is {:.1f} USD'.format(clv_basic))

'''Granular CLV calculation: CLV = Average Revenue per purchase * Average Frequency * Average Customer Lifespan'''
# Calculate average revenue per invoice
revenue_per_purchase = online.groupby(['InvoiceNo'])['TotalSum'].mean().mean()
# Calculate average number of unique invoices per customer per month
freq = online.groupby(['CustomerID', 'InvoiceMonth'])['InvoiceNo'].nunique().mean()
# Define lifespan to 36 months
lifespan_months = 36
# Calculate granular CLV
clv_granular = revenue_per_purchase * freq * lifespan_months
# Print granular CLV value
print('Average granular CLV is {:.1f} USD'.format(clv_granular))
print('Average Revenue per purchase: {:.1f} USD'.format(revenue_per_purchase))
print('Average Frequency per month: {:.1f}'.format(freq))

'''Traditional CLV calculation: CLV = Average Revenue * [ Average Retention Rate / Average Churn Rate ] '''
# Calculate monthly spend per customer
monthly_revenue = online.groupby(['CustomerID', 'InvoiceMonth'])['TotalSum'].sum().mean()
# Calculate average monthly retention rate
retention_rate = retention.iloc[:, 1:].mean().mean()
# Calculate average monthly churn rate
churn_rate = 1 - retention_rate
# Calculate traditional CLV
clv_traditional = monthly_revenue * (retention_rate / churn_rate)
# Print traditional CLV and the retention rate values
print('Average traditional CLV is {:.1f} USD at {:.1f} % retention_rate'.format(clv_traditional, retention_rate*100))
print('Average monthly Revenue: {:.1f} USD'.format(monthly_revenue))

'''Which method to use? Depends on the business model.
- Traditional CLV model assumes churn is definive = customer "dies" and does not robust at low retention values -> will under-report the CLV.
- Hardest thing to predict - frequency in the future.'''

''' Regression - predicting continuous variable. Simplest version - linear regression. Count data (e.g. number of days active) sometimes better predicted by Poisson or Negative Binomial regression '''

'''Recency, frequency, monetary (RFM) features
Recency - time since last customer transaction
Frequency - number of purchases in the observed period
Monetary value - total amount spent in the observed period
'''
# Explore the sales distribution by month
print(online.groupby(['InvoiceMonth']).size())

# Separate feature data
# Exclude target variable
online_X = online[online['InvoiceMonth']!='2011-11']
# Define snapshot date
NOW = dt.datetime(2011, 11, 1)
# Build the features
features = online_X.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW - x.max()).days,
                                               'InvoiceNo': pd.Series.nunique,
                                               'TotalSum': np.sum,
                                               'Quantity': ['mean', 'sum']}).reset_index()
features.columns = ['CustomerID', 'recency', 'frequency', 'monetary', 'quantity_avg', 'quantity_total']
print(features.head())

# Build pivot table with monthly transactions per customer
cust_month_tx = pd.pivot_table(data=online, index=['CustomerID'],
                               values='InvoiceNo',
                               columns=['InvoiceMonth'],
                               aggfunc=pd.Series.nunique, fill_value=0)
print(cust_month_tx.head())

# Finalize data preparation and split to train/test
# Store identifier and target variable column names
custid = ['CustomerID']
target = ['2011-11']
# Extract target variable
Y = cust_month_tx[target]
# Extract feature column names
cols = [col for col in features.columns if col not in custid]
# Store features
X = features[cols]

# Split data to training and testing
# Randomly split 25% of the data to testing
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.25, random_state=99)
# Print shapes of the datasets
print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)

# Predicting customer transactions
'''Linear regression to predict next month's transactions. Same modeling steps as with logistic regression.
2. Initialize the model
3. Fit the model on the training data
4. Predict values on the testing data
5. Measure model performance on testing data
'''
'''Regression performance metrics: 
1. Root mean squared error (RMSE) - Square root of the average squared difference between prediction and actuals.
2. Mean absolute error (MAE) - Average absolute difference between prediction and actuals
3. Mean absolute percentage error (MAPE) - Average percentage difference between prediction 
and actuals (actuals can't be zeros).

Additional regression and supervised learning metrics: 
5. R-squared - statistical measure that represents the percentage proportion of variance that is explained by the model.
Only applicable to regression, not classification. Higher is better! 
6. Coefficient P-value - probability that the regression (or classification) coefficient is observed due to chance. 
Lower is better! Typical thresholds are 5% and 10%.
'''

# Initialize the regression instance
linreg = LinearRegression()
# Fit model on the training data
linreg.fit(train_X, train_Y)
# Predict values on both training and testing data
train_pred_Y = linreg.predict(train_X)
test_pred_Y = linreg.predict(test_X)

# Measuring model performance
# Import performance measurement functions
# Calculate metrics for training data
rmse_train = np.sqrt(mean_squared_error(train_Y, train_pred_Y))
mae_train = mean_absolute_error(train_Y, train_pred_Y)
# Calculate metrics for testing data
rmse_test = np.sqrt(mean_squared_error(test_Y, test_pred_Y))
mae_test = mean_absolute_error(test_Y, test_pred_Y)
# Print performance metrics
print('RMSE train: {:.3f}; RMSE test: {:.3f}\nMAE train: {:.3f}, MAE test: {:.3f}'.format(
    rmse_train, rmse_test, mae_train, mae_test))

# Interpreting coefficience ->  Need to assess statistical significance
# Build regression model with statsmodels
# Convert target variable to `numpy` array
train_Y = np.array(train_Y)
# Initialize and fit the model
olsreg = sm.OLS(train_Y, train_X)
olsreg = olsreg.fit()
# Print model summary
print(olsreg.summary())

