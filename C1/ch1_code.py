import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load initial file
marketing = pd.read_csv('marketing.csv')
#marketing = pd.read_csv("https://raw.githubusercontent.com/GuoweiYang19891101/datacamp_marketing_analytics_python/main/marketing.csv")
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
