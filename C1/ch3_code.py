import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Read date columns using parse_dates
marketing = pd.read_csv('marketing.csv', parse_dates=['date_served', 'date_subscribed', 'date_canceled'])

# Change the data type of a column to a boolean
marketing['converted'] = marketing['converted'].astype('bool')
# Creating new boolean columns
marketing['is_house_ads'] = np.where(marketing['marketing_channel'] == 'House Ads', True, False)
# Mapping values to existing columns
channel_dict = {"House Ads": 1, "Instagram": 2, "Facebook": 3, "Email": 4, "Push": 5}
marketing['channel_code'] = marketing['marketing_channel'].map(channel_dict)
# Convert already existing column to datetime column
marketing['date_served'] = pd.to_datetime(marketing['date_served'])

# # Add a DoW column
marketing['DoW'] = marketing['date_subscribed'].dt.dayofweek

'''How many users see marketing assets?'''
# Aggregate unique users that see ads by date
daily_users = marketing.groupby(['date_served'])['user_id'].nunique()

'''
# Visualizing results
# Plot
daily_users.plot()
# Annotate
plt.title('Daily number of users who see ads')
plt.xlabel('Date')
plt.ylabel('Number of users')
plt.xticks(rotation=45)
# plt.show()
'''

'''Calculating conversion rate'''
# Calculate the number of people we marketed to
total = marketing['user_id'].nunique()
# Calculate the number of people who subscribed
subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
# Calculate the conversion rate
conversion_rate = subscribers/total
print('Conversion rate:',  round(conversion_rate*100, 2), '%')

'''Calculating retention rate'''
# Calculate the number of subscribers
total_subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
# Calculate the number of people who remained subscribed
retained = marketing[marketing['is_retained'] == True]['user_id'].nunique()
# Calculate the retention rate
retention_rate = retained/total_subscribers
print('Retention rate:', round(retention_rate*100, 2), "%")

'''Comparing language conversion rate '''
# Isolate english speakers
english_speakers = marketing[marketing['language_displayed'] == 'English']
# Calculate the total number of english speaking users
total = english_speakers['user_id'].nunique()
# Calculate the number of english speakers who converted
subscribers = english_speakers[english_speakers['converted'] == True]['user_id'].nunique()
# Calculate conversion rate
conversion_rate_eng = subscribers/total
print('English speaker Conversion rate:',  round(conversion_rate_eng*100,2), '%')

'''Comparing language conversion rate '''
# Group by language_displayed and count unique users
total = marketing.groupby(['language_displayed'])['user_id'].nunique()
# Group by language_displayed and count unique conversions
subscribers = marketing[marketing['converted'] == True].groupby(['language_displayed'])['user_id'].nunique()
# Calculate the conversion rate for all languages
language_conversion_rate = subscribers/total
print('Language Conversion rate:',  round(language_conversion_rate*100,2), '%')

# Group by date_served and count unique users
total = marketing.groupby(['date_served'])['user_id'].nunique()
# Group by date_served and count unique converted users
subscribers = marketing[marketing['converted'] == True].groupby(['date_served'])['user_id'].nunique()
# Calculate the conversion rate per day
daily_conversion_rate = subscribers/total
print('Daily Conversion rate:',  round(daily_conversion_rate*100,2), '%')

# Segmenting using pandas
# Subset to include only House Ads
house_ads = marketing[marketing['subscribing_channel'] == 'House Ads']
retained = house_ads[house_ads['is_retained'] == True]['user_id'].nunique()
subscribers = house_ads[house_ads['converted'] == True]['user_id'].nunique()
retention_rate = retained/subscribers
print('Retention rate only for House Ads:',  round(retention_rate*100,2), '%')

# Segmenting using pandas - groupby()
# Group by subscribing_channel and calculate retention
retained = marketing[marketing['is_retained'] == True].groupby(['subscribing_channel'])['user_id'].nunique()
# Group by subscribing_channel and calculate subscribers
subscribers = marketing[marketing['converted'] == True].groupby(['subscribing_channel'])['user_id'].nunique()
# Calculate the retention rate across the DataFrame
channel_retention_rate = (retained/subscribers)*100
print('Channel Retention rate:',  round(channel_retention_rate*100,2), '%')

'''
# Comparing language conversion rates
# Create a bar chart using channel retention DataFrame
language_conversion_rate.plot(kind = 'bar')
# Add a title and x and y-axis labels
plt.title('Conversion rate by language\n', size = 16)
plt.xlabel('Language', size = 14)
plt.ylabel('Conversion rate (%)', size = 14)
# Display the plot
# plt.show()
'''

# Group by language_displayed and count unique users
total = marketing.groupby(['date_subscribed'])['user_id'].nunique()
# Group by language_displayed and sum conversions
retained = marketing[marketing['is_retained'] == True].groupby(['date_subscribed'])\
['user_id'].nunique()
# Calculate subscriber quality across dates
daily_retention_rate = retained/total


# Reset index to turn the results into a DataFrame
daily_conversion_rate = pd.DataFrame(daily_conversion_rate.reset_index())
# Rename columns
daily_conversion_rate.columns = ['date_subscribed', 'conversion_rate']
# Create a line chart using daily_conversion_rate DataFrame
daily_conversion_rate.plot('date_subscribed', 'conversion_rate')
'''
# Add a title and x and y-axis labels
plt.title('Daily conversion rate\n', size = 16)
plt.ylabel('Conversion rate (%)', size = 14)
plt.xlabel('Date', size=14)
# Set the y-axis to begin at 0
plt.ylim(0)
# Display the plot
# plt.show()
'''

# Calculating subscriber quality
# Reset index to turn the Series into a DataFrame
daily_retention_rate = pd.DataFrame(daily_retention_rate.reset_index())
# Rename columns
daily_retention_rate.columns = ['date_subscribed', 'retention_rate']
# Create a line chart using the daily_retention DataFrame
daily_retention_rate.plot('date_subscribed', 'retention_rate')
'''
# Add a title and x and y-axis labels
plt.title('Daily subscriber quality\n', size=16)
plt.ylabel('1-month retention rate (%)', size=14)
plt.xlabel('Date', size=14)
# Set the y-axis to begin at 0
plt.ylim(0)
# Display the plot
plt.show()
'''

# Grouping by multiple columns
language = marketing.groupby(['date_served', 'language_preferred'])['user_id'].count()
# print(language.head())
# Unstacking after groupby
language = pd.DataFrame(language.unstack(level=1))
# print(language.head())

'''
# Plotting preferred language over time
language.plot()
plt.title('Daily language preferences')
plt.xlabel('Date')
plt.ylabel('Users')
plt.legend(loc = 'upper right', labels = language.columns.values)
plt.show()
'''

# Creating grouped bar charts
# Create DataFrame grouped by age and language preference
language_age = marketing.groupby(['language_preferred', 'age_group'])['user_id'].count()
language_age = pd.DataFrame(language_age.unstack(level=1))
# print(language_age.head())

'''
# Plotting language preferences by age group
language_age.plot(kind='bar')
plt.title('Language preferences by age group')
plt.xlabel('Language')
plt.ylabel('Users')
plt.legend(loc = 'upper right', labels = language_age.columns.values)
plt.show()
'''

channel_age = marketing.groupby(['marketing_channel', 'age_group'])['user_id'].count()
# Unstack channel_age and transform it into a DataFrame
channel_age_df = pd.DataFrame(channel_age.unstack(level = 1))
'''
# Plot the results
channel_age_df.plot(kind = 'bar')
plt.title('Marketing channels by age group')
plt.xlabel('Age Group')
plt.ylabel('Users')
# Add a legend to the plot
plt.legend(loc = 'upper right', labels = channel_age_df.columns.values)
# plt.show()
'''

# Count the subs by subscribing channel and date subscribed
retention_total = marketing.groupby(['date_subscribed', 'subscribing_channel'])['user_id'].nunique()
# Print results
# print(retention_total.head())

# Count the retained subs by subscribing channel and date subscribed
retention_subs = marketing[marketing['is_retained'] == True].groupby(['date_subscribed', 'subscribing_channel'])['user_id'].nunique()
# Print results
# print(retention_subs.head())

# Divide retained subscribers by total subscribers
retention_rate = retention_subs/retention_total
retention_rate_df = pd.DataFrame(retention_rate.unstack(level=1))
# Plot retention rate
retention_rate_df.plot()

'''
# Add a title, x-label, y-label, legend and display the plot
plt.title('Retention Rate by Subscribing Channel')
plt.xlabel('Date Subscribed')
plt.ylabel('Retention Rate (%)')
plt.legend(loc = 'upper right', labels = retention_rate_df.columns.values)
# plt.show()
'''

'''Next Part from here ______________________ '''


def conversion_rate(dataframe, column_names):
    # Total number of converted users
    column_conv = dataframe[dataframe['converted'] == True].groupby(column_names)['user_id'].nunique()
    # Total number users
    column_total = dataframe.groupby(column_names)['user_id'].nunique()
    # Conversion rate
    conversion_rate = column_conv / column_total
    # Fill missing values with 0
    conversion_rate = conversion_rate.fillna(0)
    return conversion_rate


# Building a retention function
def retention_rate(dataframe, column_names):
# Group by column_names and calculate retention
    retained = dataframe[dataframe['is_retained'] == True].groupby(column_names)['user_id'].nunique()
# Group by column_names and calculate conversion
    converted = dataframe[dataframe['converted'] == True].groupby(column_names)['user_id'].nunique()
    retention_rate = retained/converted
    return retention_rate

daily_retention = retention_rate(marketing,['date_subscribed', 'subscribing_channel'])
daily_retention = pd.DataFrame(daily_retention.unstack(level=1))
print(daily_retention.head())

# Test and visualize conversion function
# Calculate conversion rate by age_group
age_group_conv = conversion_rate(marketing, ['date_served', 'age_group'])
# Unstack and create a DataFrame
age_group_df = pd.DataFrame(age_group_conv.unstack(level = 1))
# Visualize conversion by age_group
age_group_df.plot()
plt.title('Conversion rate by age group\n', size = 16)
plt.ylabel('Conversion rate', size = 14)
plt.xlabel('Age group', size = 14)
plt.show()

def plotting_conv(dataframe):
    for column in dataframe:
        # Plot column by dataframe's index
        plt.plot(dataframe.index, dataframe[column])
        plt.title('Daily ' + str(column) + ' conversion rate\n', size = 16)
        plt.ylabel('Conversion rate', size = 14)
        plt.xlabel('Date', size = 14)
        # Show plot
        plt.show()
        plt.clf()

# Plotting function
def plotting(dataframe):
    for column in dataframe:
        plt.plot(dataframe.index, dataframe[column])
        plt.title('Daily ' + column + ' retention rate\n', size = 16)
        plt.ylabel('Retention rate (%)', size = 14)
        plt.xlabel('Date', size = 14)
        plt.show()

plotting(daily_retention)


# Calculate conversion rate by date served and age group
age_group_conv = conversion_rate(marketing, ['date_served', 'age_group'])

# Unstack age_group_conv and create a DataFrame
age_group_df = pd.DataFrame(age_group_conv.unstack(level=1))

# Plot the results
plotting_conv(age_group_df)

# Day of week trends
DoW_retention = retention_rate(marketing, ['DoW'])
# Plot retention by day of week
DoW_retention.plot()
plt.title('Retention rate by day of week')
plt.ylim(0)
plt.show()

# Calculate conversion rate by date served and channel
daily_conv_channel = conversion_rate(marketing, ['date_served', 'marketing_channel'])
print(daily_conv_channel.head())
# Calculate conversion rate by date served and channel
daily_conv_channel = conversion_rate(marketing, ['date_served',
                                                 'marketing_channel'])
# Unstack daily_conv_channel and convert it to a DataFrame
daily_conv_channel = pd.DataFrame(daily_conv_channel.unstack(level=1))
# Plot results of daily_conv_channel
plotting_conv(daily_conv_channel)


'''Analyzing House ads conversion rate'''
# Add day of week column to marketing
marketing['DoW_served'] = marketing['date_served'].dt.dayofweek
# Calculate conversion rate by day of week
DoW_conversion = conversion_rate(marketing, ['DoW_served', 'marketing_channel'])
# Unstack channels
DoW_df = pd.DataFrame(DoW_conversion.unstack(level=1))

# Plot conversion rate by day of week
DoW_df.plot()
plt.title('Conversion rate by day of week\n')
plt.ylim(0)
plt.show()

'''House ads conversion by language'''
# Isolate the rows where marketing channel is House Ads
house_ads = marketing[marketing['marketing_channel'] == 'House Ads']
# Calculate conversion by date served and language displayed
conv_lang_channel = conversion_rate(house_ads, ['date_served', 'language_displayed'])
# Unstack conv_lang_channel
conv_lang_df = pd.DataFrame(conv_lang_channel.unstack(level=1))
# Use plotting function to display results
plotting_conv(conv_lang_df)

'''Creating a DataFrame for house ads'''
# Add the new column is_correct_lang
house_ads['is_correct_lang'] = np.where(house_ads['language_preferred'] == house_ads['language_displayed'], 'Yes', 'No')
# Groupby date_served and is_correct_lang
language_check = house_ads.groupby(['date_served','is_correct_lang'])['is_correct_lang'].count()
# Unstack language_check and fill missing values with 0's
language_check_df = pd.DataFrame(language_check.unstack(level=1)).fillna(0)
# Print results
print(language_check_df)

'''Confirming house ads error'''
# Now that you've created a DataFrame that checks whether users see ads in the correct language
# let's calculate what percentage of users were not being served ads in the right language and plot your results.

# Divide the count where language is correct by the row sum
language_check_df['pct'] = language_check_df['Yes']/language_check_df.sum(axis=1)
# Plot and show your results
plt.plot(language_check_df.index.values, language_check_df['pct'])
plt.show()
'''Conclusion: ads have been underperforming due to serving all ads in English rather than each user's preferred language'''

'''Assessing impact'''
# Calculate pre-error conversion rate
# Bug arose sometime around '2018-01-11'
house_ads_no_bug = house_ads[house_ads['date_served'] < '2018-01-11']
lang_conv = conversion_rate(house_ads_no_bug, ['language_displayed'])

# Index other language conversion rate against English
spanish_index = lang_conv['Spanish']/lang_conv['English']
arabic_index = lang_conv['Arabic']/lang_conv['English']
german_index = lang_conv['German']/lang_conv['English']

print("Spanish index:", spanish_index)
print("Arabic index:", arabic_index)
print("German index:", german_index)

'''The same :
# Calculate pre-error conversion rate
house_ads_bug = house_ads[house_ads['date_served'] < '2018-01-11']
lang_conv = conversion_rate(house_ads_bug, ['language_displayed'])

# Index other language conversion rate against English
spanish_index = lang_conv['Spanish']/lang_conv['English']
arabic_index = lang_conv['Arabic']/lang_conv['English']
german_index = lang_conv['German']/lang_conv['English']

print("Spanish index:", spanish_index)
print("Arabic index:", arabic_index)
print("German index:", german_index)
'''

# Daily conversion
# Create actual conversion DataFrame
language_conversion = house_ads.groupby(['date_served', 'language_preferred']).agg({'user_id': 'nunique', 'converted': 'sum'})

expected_conversion = pd.DataFrame(language_conversion.unstack(level=1))
print(expected_conversion)

# Create English conversion rate column for affected period
language_conversion['actual_english_conversions'] = language_conversion.loc['2018-01-11':'2018-01-31'][('converted', 'English')]

''' ===========================[!] KeyError: ('converted', 'English')'''

# Calculating daily expected conversion rate
# Create expected conversion rates for each language
language_conversion['expected_spanish_rate'] = language_conversion['actual_english_rate']*spanish_index
language_conversion['expected_arabic_rate'] = language_conversion['actual_english_rate']*arabic_index
language_conversion['expected_german_rate'] = language_conversion['actual_english_rate']*german_index

# Multiply total ads served by expected conversion rate
language_conversion['expected_spanish_conversions'] = language_conversion['expected_spanish_rate']/100*language_conversion[('user_id', 'Spanish')]
language_conversion['expected_arabic_conversions'] = language_conversion['expected_arabic_rate']/100*language_conversion[('user_id', 'Arabic')]
language_conversion['expected_german_conversions'] = language_conversion['expected_german_rate']/100*language_conversion[('user_id', 'German')]

bug_period = language_conversion.loc['2018-01-11':'2018-01-31']
# Sum expected subscribers for each language
expected_subs = bug_period['expected_spanish_conv_rate'].agg('sum')
bug_period['expected_arabic_conv_rate'].agg('sum') + bug_period['expected_german_conv_rate'].agg('sum')
# Calculate how many subscribers we actually got
actual_subs = bug_period[('converted', 'Spanish')].sum() + bug_period[('converted', 'Arabic')].agg('sum') + bug_period[('converted', 'German')].agg('sum')
lost_subs = expected_subs - actual_subs
print(lost_subs)