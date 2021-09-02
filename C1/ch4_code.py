import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read date columns using parse_dates
marketing = pd.read_csv('marketing_new.csv', parse_dates=['date_served', 'date_subscribed', 'date_canceled'])

'''Test allocation - first we need to check control group VS testing group'''
# Subset the DataFrame
email = marketing[marketing['marketing_channel'] == 'Email']
# Group the email DataFrame by variant
allocation = email.groupby(['variant'])['user_id'].nunique()
# Plot a bar chart of the test allocation
allocation.plot(kind='bar')
plt.title('Personalization test allocation')
plt.ylabel('# participants')
plt.show()

'''Comparing conversion rates of Testing group and Control group'''
# Group the email DataFrame by user_id and variant while selecting the maximum
# value of the converted column and store the results in subscribers.
subscribers = email.groupby(['user_id', 'variant'])['converted'].max()
subscribers_df = pd.DataFrame(subscribers.unstack(level=1))
# Drop missing values from the control column
control = subscribers_df['control'].dropna()
# Drop missing values from the personalization column
personalization = subscribers_df['personalization'].dropna()
print('Control conversion rate:', np.mean(control))
print('Personalization conversion rate:', np.mean(personalization))

'''Treatment performance compared to the control
Calculating lift: [ Treatment conversion rate - Control conversion rate ]  /  Control conversion rate'''

# Calculating lift
# Calcuate the mean of a and b
a_mean = np.mean(control)
b_mean = np.mean(personalization)
# Calculate the lift using a_mean and b_mean
lift = (b_mean-a_mean)/a_mean
print("lift:", str(round(lift*100, 2)) + '%')

# T-distribution, t-test and P-values (http://bytepawn.com/ab-testing-and-the-ttest.html)
from scipy.stats import ttest_ind, stats

t = ttest_ind(control, personalization)
print(t)
# A p-value less than 0.05 (typically ≤ 0.05) is statistically significant.
# It indicates strong evidence against the null hypothesis,
# as there is less than a 5% probability the null is correct (and the results are random) or the result is 95% significant.

# Building an A/B test segmenting function
def ab_segmentation(segment):
    # Build a for loop for each segment in marketing
    for subsegment in marketing[segment].values:
        print(subsegment)

        # Limit marketing to email and subsegment
        email = marketing[(marketing['marketing_channel'] == 'Email') & (marketing[segment] == subsegment)]

        subscribers = email.groupby(['user_id', 'variant'])['converted'].max()
        subscribers = pd.DataFrame(subscribers.unstack(level=1))
        control = subscribers['control'].dropna()
        personalization = subscribers['personalization'].dropna()

        print('lift:', lift(control, personalization))
        print('t-statistic:', stats.ttest_ind(control, personalization), '\n\n')

# Use ab_segmentation on language displayed
ab_segmentation('language_displayed')

''' ===========================[!] TypeError: 'numpy.float64' object is not callable'''

# Use ab_segmentation on age group
ab_segmentation('age_group')
