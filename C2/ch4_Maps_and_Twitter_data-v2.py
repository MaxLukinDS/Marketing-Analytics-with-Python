import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# Import Basemap
from mpl_toolkits.basemap import Basemap

# open JSON file
tweet_json_3 = open('tweet_json 3.json', 'r').read()
tweets_sotu_json = open('tweets_sotu_json.json', 'r').read()

'''A tweet flattening function'''
def flatten_tweets(tweets_json):
    """ Flattens out tweet dictionaries so relevant JSON is in a top-level dictionary."""
    tweets_list = []

    # Iterate through each tweet
    for tweet in tweets_json:
        tweet_obj = json.loads(tweet)

        # Store the user screen name in 'user-screen_name'
        tweet_obj['user-screen_name'] = tweet_obj['user']['screen_name']

        # Check if this is a 140+ character tweet
        if 'extended_tweet' in tweet_obj:
            # Store the extended tweet text in 'extended_tweet-full_text'
            tweet_obj['extended_tweet-full_text'] = tweet_obj['extended_tweet']['full_text']

        if 'retweeted_status' in tweet_obj:
            # Store the retweet user screen name in 'retweeted_status-user-screen_name'
            tweet_obj['retweeted_status-user-screen_name'] = tweet_obj['retweeted_status']['user']['screen_name']

            # Store the retweet text in 'retweeted_status-text'
            tweet_obj['retweeted_status-text'] = tweet_obj['retweeted_status']['text']
            tweet_obj['user-location'] = tweet_obj['user']['location']

        tweets_list.append(tweet_obj)
    return tweets_list

'''Accessing user-defined location'''
# Print out the location of a single tweet
print(tweet_json_3['user']['location'])
'''[!] ERROR: TypeError: string indices must be integers'''

# tweet_json_df = pd.DataFrame(flatten_tweets(tweet_json))
# print(tweet_json_df['user-location'])
'''[!] ERROR: json.decoder.JSONDecodeError: Expecting property name enclosed in double quotes: line 1 column 2 (char 1)'''

# Flatten and load the SOTU tweets into a dataframe
tweets_sotu = pd.DataFrame(flatten_tweets(tweets_sotu_json))
# Print out top five user-defined locations
print(tweets_sotu['user-location'].value_counts().head())

'''Accessing bounding box:'''
'''Most tweets which have coordinate-level geographical information attached to them 
typically come in the form of a bounding box. Bounding boxes are a set of four longitudinal/latitudinal coordinates 
which denote a particular area in which the user can be located.'''
def getBoundingBox(place):
    """ Returns the bounding box coordinates."""
    return place['bounding_box']['coordinates']

# Apply the function which gets bounding box coordinates
bounding_boxes = tweets_sotu['place'].apply(getBoundingBox)
# Print out the first bounding box coordinates
print(bounding_boxes.values[0])

'''Calculating the centroid'''
'''The bounding box can range from a city block to a whole state or even country. 
For simplicity's sake, one way we can deal with handling these data is by translating the bounding box into 
what's called a centroid, or the center of the bounding box. '''
def calculateCentroid(place):
    """ Calculates the centroid from a bounding box."""
    # Obtain the coordinates from the bounding box.
    coordinates = place['bounding_box']['coordinates'][0]

    longs = np.unique([x[0] for x in coordinates])
    lats = np.unique([x[1] for x in coordinates])

    if len(longs) == 1 and len(lats) == 1:
        # return a single coordinate
        return (longs[0], lats[0])
    elif len(longs) == 2 and len(lats) == 2:
        # If we have two longs and lats, we have a box.
        central_long = np.sum(longs) / 2
        central_lat = np.sum(lats) / 2
    else:
        raise ValueError("Non-rectangular polygon not supported.")

    return (central_long, central_lat)

# Calculate the centroids of place
centroids = tweets_sotu['place'].apply(calculateCentroid)

'''Creating Basemap map'''
'''Basemap allows you to create maps in Python. The library builds projections for latitude and longitude coordinates 
and then passes the plotting work on to matplotlib.
This means you can build extra features based on the power of matplotlib.'''
# Set up the US bounding box
us_boundingbox = [-125, 22, -64, 50]

# Set up the Basemap object
m = Basemap(llcrnrlon = us_boundingbox[0],
            llcrnrlat = us_boundingbox[1],
            urcrnrlon = us_boundingbox[2],
            urcrnrlat = us_boundingbox[3],
            projection='merc')

# Draw continents in white,
# coastlines and countries in gray
m.fillcontinents(color='white')
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')

# Draw the states and show the plot
m.drawstates(color='gray')
plt.show()

'''Plotting centroid coordinates'''
# Calculate the centroids for the dataset
# and isolate longitudue and latitudes
centroids = tweets_sotu['place'].apply(calculateCentroid)
lon = [x[0] for x in centroids]
lat = [x[1] for x in centroids]

# Draw continents, coastlines, countries, and states
m.fillcontinents(color='white', zorder = 0)
m.drawcoastlines(color='gray')
m.drawcountries(color='gray')
m.drawstates(color='gray')

# Draw the points and show the plot
m.scatter(lon, lat, latlon = True, alpha = 0.7)
plt.show()

'''Coloring by sentiment'''
# Generate sentiment scores
sentiment_scores = tweets_sotu['text'].apply(sid.polarity_scores)
# Isolate the compound element
sentiment_scores = [x['compound'] for x in sentiment_scores]
# Draw the points
m.scatter(lon, lat, latlon=True, c=sentiment_scores, cmap = 'coolwarm', alpha = 0.7)
# Show the plot
plt.show()

