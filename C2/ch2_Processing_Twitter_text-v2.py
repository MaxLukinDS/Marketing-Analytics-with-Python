import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load initial file
#tweets = pd.read_csv('Data Science Hashtag dataset.csv')

# OR
tweets = pd.read_csv('Data Science Hashtag dataset.csv', parse_dates=['created_at'])
tweets['created_at'] = tweets['created_at'].dt.tz_localize(tz=None)
print(tweets.info())

'''Looking for text in all the wrong places:
Relevant text may not only be in the main text field of the tweet. It may also be in the 
- extended_tweet, 
- the retweeted_status, or 
- the quoted_status. 
We need to check all of these fields to make sure we've accounted for all the of the relevant text. '''

def check_word_in_tweet(word, data):
    """Checks if a word is in a Twitter dataset's text.
    Checks text and extended tweet (140+ character tweets) for tweets,
    retweets and quoted tweets.
    Returns a logical pandas Series.
    """

    contains_column = data['text'].str.contains(word, case=False)
    contains_column |= data['extended_tweet-full_text'].str.contains(word, case=False)
    contains_column |= data['quoted_status-text'].str.contains(word, case=False)
    contains_column |= data['quoted_status-extended_tweet-full_text'].str.contains(word, case=False)
    contains_column |= data['retweeted_status-text'].str.contains(word, case=False)
    contains_column |= data['retweeted_status-extended_tweet-full_text'].str.contains(word, case=False)

    return contains_column

'''Comparing #python to #rstats'''
# Find mentions of #python in all text fields
python = check_word_in_tweet('#python', tweets)
# Find mentions of #rstats in all text fields
rstats = check_word_in_tweet('#rstats', tweets)
# Print proportion of tweets mentioning #python
print("Proportion of #python tweets:", np.sum(python) / tweets.shape[0])
# Print proportion of tweets mentioning #rstats
print("Proportion of #rstats tweets:", np.sum(rstats) / tweets.shape[0])

'''Creating time series data frame'''
# Print created_at to see the original format of datetime in Twitter data
print(tweets['created_at'].head())
# Convert the created_at column to np.datetime object
tweets['created_at'] = pd.to_datetime(tweets['created_at'])
# Print created_at to see new format
print(tweets['created_at'].head())

# Set the index of ds_tweets to created_at
tweets = tweets.set_index('created_at')

# Create a python column
tweets['python'] = check_word_in_tweet('#python', tweets)
# Create an rstats column
tweets['rstats'] = check_word_in_tweet('#rstats', tweets)

# Average of python column by day
mean_python = tweets['python'].resample('1 d').mean()
# Average of rstats column by day
mean_rstats = tweets['rstats'].resample('1 d').mean()

'''Plotting mean frequency'''
# Plotting mean frequency
# Plot mean python/rstats by day
plt.plot(mean_python.index.day, mean_python, color = 'green')
plt.plot(mean_rstats.index.day, mean_rstats, color = 'blue')
# Add labels and show
plt.xlabel('Day')
plt.ylabel('Frequency')
plt.title('Language mentions over time')
plt.legend(('#python', '#rstats'))
plt.show()

'''✨Sentiment Analysis Method:
Counting positive/negative words VS positivity/negativity by VADER SentimentIntensityAnalyzer()
Part of Natural Language Toolkit (nltk)
Good for short texts like tweets
Measures sentiment of particular words (e.g. angry, happy)
Also considers sentiment of emoji (😅😅) and capitalization (Nice vs NICE)'''

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
#sentiment_scores = tweets['text'].apply(sid.polarity_scores)

'''
sid.polarity_scores(tweet) -> {'neg': 0.4, 'compound': 0.1, 'pos': 0.2, 'neu':0.3}
- neg: negative sentiment
- pos: positive sentiment
- neu: neutral measures words that don't contribute to the sentiment
- compound: combination of positive and negative(range [-1,1])
'''
#for i in range(len(ds_tweets)): print(sentiment_scores[i]['compound'])

#Creating a sentiment column in tweet dataframe itself
tweets['sentiment'] = 0
for i in range(len(tweets)):
    tweets.iloc[i,42] = sid.polarity_scores(tweets.iloc[i,37])['compound']

'''Calculating sentiment scores'''
# Print out the text of a positive tweet
print(tweets[tweets.sentiment > 0.6]['text'].values[0])
# Print out the text of a negative tweet
print(tweets[tweets.sentiment < -0.6]['text'].values[0])

#method 1
# Generate average sentiment scores for #python 
sentiment_py = tweets[check_word_in_tweet('#python', tweets)]['sentiment'].resample('1 d').mean()
# Generate average sentiment scores for #rstats
sentiment_r = tweets[check_word_in_tweet('#rstats', tweets)]['sentiment'].resample('1 d').mean()

#method 2
tweets['python'] = check_word_in_tweet('#python',tweets)
tweets['rstats'] = check_word_in_tweet('#rstats',tweets)
# Generate average sentiment scores for #python
sentiment_py = tweets[tweets['python']]['sentiment'].resample('1 d').mean()
# Generate average sentiment scores for #rstats
sentiment_r = tweets[tweets['rstats']]['sentiment'].resample('1 d').mean()

'''Plotting sentiment scores'''
# Plot average #python sentiment per day
plt.plot(sentiment_py.index.day, sentiment_py, color = 'green')
# Plot average #rstats sentiment per day
plt.plot(sentiment_r.index.day, sentiment_r, color = 'blue')
plt.xlabel('Day')
plt.ylabel('Sentiment')
plt.title('Sentiment of data science languages')
plt.legend(('#python', '#rstats'))
plt.show()


