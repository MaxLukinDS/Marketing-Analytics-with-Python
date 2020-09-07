import pandas as pd
import datetime as dt

'''
Behavioral customer segmentation based on three metrics:
Recency (R) -> days since last customer transaction
Frequency (F) -> number of transactions in the last 12 months
Monetary Value (M) -> total spend in the last 12 months
'''

# Load initial file
online12M = pd.read_csv('online12M.csv')
print(online12M.head())
print(online12M.info())

online12M['InvoiceDate'] = pd.to_datetime(online12M['InvoiceDate'])
# Add a new column TotalSum = Quantity x UnitPrice
online12M['TotalSum'] = online12M['Quantity'] * online12M['UnitPrice']

# To make sure that we have 1 year data
print('Min:{}; Max:{}'.format(min(online12M.InvoiceDate), max(online12M.InvoiceDate)))

# Let's create a hypothetical snapshot_day data as if we're doing analysis recently
snapshot_date = max(online12M.InvoiceDate) + dt.timedelta(days=1)

# Aggregate data on a customer level
datamart = online12M.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Rename columns for easier interpretation
datamart.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue'}, inplace=True)

# Print RFM
print(datamart.head())

# Recency quartile
'''[!] since the Recency value measure the days since last customer transaction, we will rate 
       the customers, who have been active more recently better than the least recent customers -> 
       so we will sort the customers in the increasing order from 4 to 1 '''
r_labels = range(4, 0, -1)
r_quartiles = pd.qcut(datamart['Recency'], 4, labels=r_labels)
# assign these values to the new column R
datamart = datamart.assign(R=r_quartiles.values)

# Frequency and Monetary quartiles
'''[!] Frequency and Monetary are considered better, when they are higher -> 
       so we will sort the customers in the order from 1 to 4 '''
f_labels = range(1, 5)
m_labels = range(1, 5)
f_quartiles = pd.qcut(datamart['Frequency'], 4, labels=f_labels)
m_quartiles = pd.qcut(datamart['MonetaryValue'], 4, labels=m_labels)
datamart = datamart.assign(F=f_quartiles.values)
datamart = datamart.assign(M=m_quartiles.values)
print(datamart.head())

# Build RFM Segment and RFM Score
# Concatenate RFM quartile values to RFM_Segment
def join_rfm(x): return str(x['R']) + str(x['F']) + str(x['M'])
datamart['RFM_Segment'] = datamart.apply(join_rfm, axis=1)

# Sum RFM quartiles values to RFM_Score
datamart['RFM_Score'] = datamart[['R', 'F', 'M']].sum(axis=1)
print(datamart.head())

# Analyzing RFM segments
print(datamart.groupby('RFM_Segment').size().sort_values(ascending=False)[:10])

# Filtering on RFM segments with the lowest RFM segmentation '111' and RFM_score
print(datamart[datamart['RFM_Segment'] == '111'][:5])

# Summary metrics per RFM Score
print(datamart.groupby('RFM_Score').agg({'Recency': 'mean',
                                   'Frequency': 'mean',
                                   'MonetaryValue': ['mean', 'count'] }).round(1))

# Grouping into named segments
# Use RFM score to group customers into Gold, Silver and Bronze segments.

def segment_me(df):
    if df['RFM_Score'] >= 9:
        return 'Gold'
    elif (df['RFM_Score'] >= 5) and (df['RFM_Score'] < 9):
        return 'Silver'
    else:
        return 'Bronze'

datamart['General_Segment'] = datamart.apply(segment_me, axis=1)

print(datamart.groupby('General_Segment').agg({'Recency': 'mean',
                                         'Frequency': 'mean',
                                         'MonetaryValue': ['mean', 'count']}).round(1))




