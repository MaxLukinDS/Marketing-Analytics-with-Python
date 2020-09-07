import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load initial file
marketing = pd.read_csv('marketing.csv')
# OR Read date columns using parse_dates
# marketing = pd.read_csv('marketing.csv', parse_dates=['date_served', 'date_subscribed', 'date_canceled'])

# Inspecting data
print(marketing.head())
# Summary statistics
print(marketing.describe())
# Missing values & data types
print(marketing.info())

'''Common data types:
Strings (objects)
Numbers (float, integers)
Boolean values (True, False)
Dates'''

# Print a data type of a single column
print(marketing['converted'].dtype)

# Change the data type of a column to a boolean
marketing['converted'] = marketing['converted'].astype('bool')
print(marketing['converted'].dtype)

# Creating new boolean columns
marketing['is_house_ads'] = np.where(marketing['marketing_channel'] == 'House Ads', True, False)
print(marketing.is_house_ads.head(3))

# Mapping values to existing columns
channel_dict = {"House Ads": 1, "Instagram": 2, "Facebook": 3, "Email": 4, "Push": 5}
marketing['channel_code'] = marketing['marketing_channel'].map(channel_dict)
print(marketing['channel_code'].head(3))

# Read date columns using parse_dates
# marketing = pd.read_csv('marketing.csv', parse_dates=['date_served', 'date_subscribed', 'date_canceled'])
# Or
# Convert already existing column to datetime column
marketing['date_served'] = pd.to_datetime(marketing['date_served'])

# # Add a DoW column
# marketing['DoW'] = marketing['date_subscribed'].dt.dayofweek

'''How many users see marketing assets?'''
# Aggregate unique users that see ads by date
daily_users = marketing.groupby(['date_served'])['user_id'].nunique()
print(daily_users)

# Visualizing results
# Plot
daily_users.plot()
# Annotate
plt.title('Daily number of users who see ads')
plt.xlabel('Date')
plt.ylabel('Number of users')
plt.xticks(rotation=45)
plt.show()

'''Calculating conversion rate using pandas'''
# Calculate the number of people we marketed to
total = marketing['user_id'].nunique()
# Calculate the number of people who subscribed
subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
# Calculate the conversion rate
conv_rate = subscribers/total
print('Conversion rate:',  round(conv_rate*100, 2), '%')

'''Calculating retention rate using pandas'''
# Calculate the number of subscribers
total_subscribers = marketing[marketing['converted'] == True]['user_id'].nunique()
# Calculate the number of people who remained subscribed
retained = marketing[marketing['is_retained'] == True]['user_id'].nunique()
# Calculate the retention rate
retention_rate = retained/total_subscribers
print('Retention rate:', round(retention_rate*100, 2), "%")

'''Comparing language conversion rate (I)'''
# Isolate english speakers
english_speakers = marketing[marketing['language_displayed'] == 'English']
# Calculate the total number of english speaking users
total = english_speakers['user_id'].nunique()
# Calculate the number of english speakers who converted
subscribers = english_speakers[english_speakers['converted'] == True]['user_id'].nunique()
# Calculate conversion rate
conversion_rate = subscribers/total
print('English speaker conversion rate:',  round(conversion_rate*100,2), '%')

'''Comparing language conversion rate (II)'''
# Group by language_displayed and count unique users
total = marketing.groupby(['language_displayed'])['user_id'].nunique()
# Group by language_displayed and count unique conversions
subscribers = marketing[marketing['converted'] == True].groupby(['language_displayed'])['user_id'].nunique()
# Calculate the conversion rate for all languages
language_conversion_rate = subscribers/total
print(language_conversion_rate)

# Group by date_served and count unique users
total = marketing.groupby(['date_served'])['user_id'].nunique()
# Group by date_served and count unique converted users
subscribers = marketing[marketing['converted'] == True].groupby(['date_served'])['user_id'].nunique()
# Calculate the conversion rate per day
daily_conversion_rate = subscribers/total
print(daily_conversion_rate)

# Segmenting using pandas
# Subset to include only House Ads
house_ads = marketing[marketing['subscribing_channel'] == 'House Ads']
retained = house_ads[house_ads['is_retained'] == True]['user_id'].nunique()
subscribers = house_ads[house_ads['converted'] == True]['user_id'].nunique()
retention_rate = retained/subscribers
print(round(retention_rate*100, 2), '%')

# Segmenting using pandas - groupby()
# Group by subscribing_channel and calculate retention
retained = marketing[marketing['is_retained'] == True].groupby(['subscribing_channel'])['user_id'].nunique()
print(retained)
# Group by subscribing_channel and calculate subscribers
subscribers = marketing[marketing['converted'] == True].groupby(['subscribing_channel'])['user_id'].nunique()
print(subscribers)
# Calculate the retention rate across the DataFrame
channel_retention_rate = (retained/subscribers)*100
print(channel_retention_rate)

# Comparing language conversion rates
# Create a bar chart using channel retention DataFrame
language_conversion_rate.plot(kind = 'bar')
# Add a title and x and y-axis labels
plt.title('Conversion rate by language\n', size = 16)
plt.xlabel('Language', size = 14)
plt.ylabel('Conversion rate (%)', size = 14)
# Display the plot
plt.show()

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
# Add a title and x and y-axis labels
plt.title('Daily conversion rate\n', size = 16)
plt.ylabel('Conversion rate (%)', size = 14)
plt.xlabel('Date', size=14)
# Set the y-axis to begin at 0
plt.ylim(0)
# Display the plot
plt.show()


# Calculating subscriber quality
# Reset index to turn the Series into a DataFrame
daily_retention_rate = pd.DataFrame(daily_retention_rate.reset_index())
# Rename columns
daily_retention_rate.columns = ['date_subscribed', 'retention_rate']
# Create a line chart using the daily_retention DataFrame
daily_retention_rate.plot('date_subscribed', 'retention_rate')
# Add a title and x and y-axis labels
plt.title('Daily subscriber quality\n', size=16)
plt.ylabel('1-month retention rate (%)', size=14)
plt.xlabel('Date', size=14)
# Set the y-axis to begin at 0
plt.ylim(0)
# Display the plot
plt.show()

# Grouping by multiple columns
language = marketing.groupby(['date_served', 'language_preferred'])['user_id'].count()
print(language.head())
# Unstacking after groupby
language = pd.DataFrame(language.unstack(level=1))
print(language.head())

# Plotting preferred language over time
language.plot()
plt.title('Daily language preferences')
plt.xlabel('Date')
plt.ylabel('Users')
plt.legend(loc = 'upper right', labels = language.columns.values)
plt.show()

# Creating grouped bar charts
# Create DataFrame grouped by age and language preference
language_age = marketing.groupby(['language_preferred', 'age_group'])['user_id'].count()
language_age = pd.DataFrame(language_age.unstack(level=1))
print(language_age.head())

# Plotting language preferences by age group
language_age.plot(kind='bar')
plt.title('Language preferences by age group')
plt.xlabel('Language')
plt.ylabel('Users')
plt.legend(loc = 'upper right', labels = language_age.columns.values)
plt.show()

channel_age = marketing.groupby(['marketing_channel', 'age_group'])['user_id'].count()
# Unstack channel_age and transform it into a DataFrame
channel_age_df = pd.DataFrame(channel_age.unstack(level = 1))

# Plot the results
channel_age_df.plot(kind = 'bar')
plt.title('Marketing channels by age group')
plt.xlabel('Age Group')
plt.ylabel('Users')
# Add a legend to the plot
plt.legend(loc = 'upper right', labels = channel_age_df.columns.values)
plt.show()

# Count the subs by subscribing channel and date subscribed
retention_total = marketing.groupby(['date_subscribed', 'subscribing_channel'])['user_id'].nunique()
# Print results
print(retention_total.head())

# Count the retained subs by subscribing channel and date subscribed
retention_subs = marketing[marketing['is_retained'] == True].groupby(['date_subscribed', 'subscribing_channel'])['user_id'].nunique()
# Print results
print(retention_subs.head())


# Divide retained subscribers by total subscribers
retention_rate = retention_subs/retention_total
retention_rate_df = pd.DataFrame(retention_rate.unstack(level=1))

# Plot retention rate
retention_rate_df.plot()

# Add a title, x-label, y-label, legend and display the plot
plt.title('Retention Rate by Subscribing Channel')
plt.xlabel('Date Subscribed')
plt.ylabel('Retention Rate (%)')
plt.legend(loc = 'upper right', labels = retention_rate_df.columns.values)
plt.show()



