import numpy as np
import pandas as pd
from datetime import timedelta

# Load initial files
customer_data = pd.read_csv('customer_data.csv')
app_purchases = pd.read_csv('app_purchases.csv')
# Print the columns of customer_data
print(customer_data.info())
print(customer_data.columns)
# Print the columns of app_purchases
print(app_purchases.info())
print(app_purchases.columns)

customer_data['reg_date'] = pd.to_datetime(customer_data['reg_date'])
# With the tz=None, we can remove the timezone
customer_data['reg_date'] = customer_data['reg_date'].dt.tz_localize(tz=None)
app_purchases['date'] = pd.to_datetime(app_purchases['date'])
# Merge on the 'uid' field
purchase_data = app_purchases.merge(customer_data, on=['uid'], how='inner')
purchase_data.rename(columns={"date": "subscription_date"}, inplace=True)

# Examine the results
print(purchase_data.head())
print(app_purchases.columns)

''' Визуализировать всех клиентов!
1. Кол-во новых юзеров по месяцам 
2. прирост их по дням 
'''

# Calculate the mean purchase price
purchase_price_mean = purchase_data.price.agg('mean')
print(f"Mean purchase price:{purchase_price_mean:.2f}")
# Group the data
grouped_purchase_data = purchase_data.groupby(by=['gender', 'device'])
# aggregate info about subscription price (mean, median, std):
purchase_summary1 = grouped_purchase_data.agg({'price': ['mean', 'median', 'std']})
# Group the data
grouped_purchase_data = purchase_data.groupby(by=['country', 'gender', 'device'])
purchase_summary2 = grouped_purchase_data.agg({'price': ['mean', 'median', 'min', 'max'],
                                                       'age': ['mean', 'median', 'min', 'max']})

grouped_purchase_data = purchase_data.groupby(by=['country', 'age', ])
purchase_summary3 = grouped_purchase_data.agg({'price': ['mean', 'median'],
                                               'uid': ['nunique']})

print("Purchase statistics (mean, median, std) by device and gender.")
print(purchase_summary1)
print("Purchase statistics (mean, median, std) by country, device and gender.")
print(purchase_summary2)
print("Purchase statistics (mean, median, std) by country and age.")
print(purchase_summary3)

# The maximum date in our dataset
current_date = pd.to_datetime('2018-03-17')  # timestamp


def conversion_rate_for_n_days (n):
    """The function will calculate for specified number of days period:
    1) Total number of Users, 2) Number of Users, who subscribed within the first n days,
    3) The average amount paid per purchase within user's first n days,
    4) Conversion Rate for users who subscribed within the first n days.
    Args:
        n: specified number of days
    Returns:
        will print the string with 4 calculated parameters """

    # Compute max_purchase_date
    max_purchase_date = current_date - timedelta(days=n)
    # Filter to only include users who registered before our max date
    purchase_data_filt = purchase_data[purchase_data.reg_date < max_purchase_date]
    # Filter to contain only purchases within the first N days of registration
    purchase_data_filt = purchase_data_filt[(purchase_data_filt.subscription_date
                                             <= purchase_data_filt.reg_date + timedelta(days=n))]
    # Output the mean price paid per purchase
    mean_price_within_n_days = purchase_data_filt.price.mean()

    # restrict to users lapsed before max_lapse_date
    conv_sub_data = purchase_data[(purchase_data.reg_date < max_purchase_date)]
    # count the users remaining in our data
    total_users_count = conv_sub_data.price.count()
    # latest subscription date: within N days of lapsing
    max_sub_date = conv_sub_data.reg_date + timedelta(days=n)
    # filter the users with non-zero subscription price and who subscribed before max_sub_date
    total_subs = conv_sub_data[(conv_sub_data.price > 0) & (conv_sub_data.subscription_date <= max_sub_date)]
    # count the users remaining in our data
    total_subs_count = total_subs.price.count()
    # calculate the conversion rate with our previous values
    conversion_rate_n_days = total_subs_count / total_users_count

    return print("Total number of Users: {},\n"
                 "Number of Users, who subscribed within the first {} days :{:.0f},\n"
                 "The average amount paid per purchase within user's first {} days: {:.2f},\n"
                 "Conversion Rate for users who subscribed within the first {} days: {:.6f}."
                 .format(total_users_count, n, total_subs_count, n, mean_price_within_n_days, n, conversion_rate_n_days))

def avg_cohort_purchase_for_n_days (n):
    """The function will do the following for specified number of days period:
        1) Filter and keep only accurate purchases,
        3) Group the data by gender and device,
        4) Aggregate data using the required period and price data.
        Args:
            n: specified number of days
        Returns:
            will return a final table after grouping and aggregation of data """

    # Compute max_purchase_date
    max_purchase_date = current_date - timedelta(days=n)
    # Find the n period values
    period = np.where((purchase_data.reg_date < max_purchase_date) & (
            purchase_data.subscription_date < purchase_data.reg_date + timedelta(days=n)),
                     purchase_data.price, np.NaN)
    # Update the value in the DataFrame
    purchase_data['period'] = period
    # Group the data by gender and device
    purchase_data_upd = purchase_data.groupby(by=['gender', 'device'], as_index=False)
    # Aggregate the required period and price data
    purchase_summary = purchase_data_upd.agg({'period': ['mean', 'median'], 'price': ['mean', 'median']})

    return print(purchase_summary)

conversion_rate_for_n_days(7)
avg_cohort_purchase_for_n_days(7)

conversion_rate_for_n_days(28)
avg_cohort_purchase_for_n_days(28)

'''Cohort conversion rate'''
# Lapse Date: Date the trial ends for a given user
# purchase_cohorts = conv_sub_data.groupby(by=['gender', 'device'], as_index=False)
# purchase_summary4 = purchase_cohorts.agg({sub_time: [gcr7,gcr14]})

