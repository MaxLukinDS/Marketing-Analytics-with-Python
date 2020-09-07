import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Load initial file
online = pd.read_csv('online.csv')

print(online.head())
print(online.info())

online['InvoiceDate'] = pd.to_datetime(online['InvoiceDate'])

print(online.info())

# Assign acquisition month cohort
def get_month(x): return pd.datetime(x.year, x.month, 1)


online['InvoiceMonth'] = online['InvoiceDate'].apply(get_month)
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

# Calculate Retention rate
# Store the first column as cohort_sizes
cohort_sizes = cohort_counts.iloc[:, 0]
# Divide all values in the cohort_counts table by cohort_sizes
retention = cohort_counts.divide(cohort_sizes, axis=0)
# Review the retention table
print(retention.round(3) * 100)

# Other metrics:
# Average quantity for each cohort
grouping = online.groupby(['CohortMonth', 'CohortIndex'])
# Calculate the average quantity
cohort_data = grouping['Quantity'].mean()
# Reset the index of cohort_data
cohort_data = cohort_data.reset_index()
# Create a pivot
average_quantity = cohort_data.pivot(index='CohortMonth',
                                     columns='CohortIndex',
                                     values='Quantity')
print(average_quantity.round(1))

# Average price for each cohort
# Create a groupby object and pass the monthly cohort and cohort index as a list
grouping = online.groupby(['CohortMonth', 'CohortIndex'])
# Calculate the average of the unit price
cohort_data = grouping['UnitPrice'].mean()
# Reset the index of cohort_data
cohort_data = cohort_data.reset_index()
# Create a pivot
average_price = cohort_data.pivot(index='CohortMonth',
                                  columns='CohortIndex',
                                  values='UnitPrice')
print(average_price.round(1))

# Cohort analysis visualization - Heatmap
plt.figure(figsize=(10, 8))
plt.title('Retention rates')
sns.heatmap(data=retention,
            annot=True,
            fmt='.0%',
            vmin=0.0,
            vmax=0.5,
            cmap='BuGn')
plt.show()
