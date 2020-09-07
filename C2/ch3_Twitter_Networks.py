# Importing a retweet network
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Load initial file
sotu_retweets = pd.read_csv('State of the Union Retweet Networking dataset.csv')
sotu_replies = pd.read_csv('State of the Union Reply Network dataset.csv')

'''Creating retweet network'''
# Create retweet network from edgelist
G_rt = nx.from_pandas_edgelist(sotu_retweets,
                               source='user-screen_name',
                               target='retweeted_status-user-screen_name',
                               create_using=nx.DiGraph())
# Print the number of nodes
print('Nodes in RT network:', len(G_rt.nodes()))
# Print the number of edges
print('Edges in RT network:', len(G_rt.edges()))

'''Visualizing retweet network'''
# Create random layout positions
pos = nx.random_layout(G_rt)
# Create size list
sizes = [x[1] for x in G_rt.degree()]
# Draw the network
nx.draw_networkx(G_rt, pos, with_labels = False,
                 node_size =sizes,
                 width =0.1, alpha=0.7,
                 arrowsize =2, linewidths=0)
# Turn axis off and show
plt.axis('off')
plt.show()

'''Creating reply network'''
# Create reply network from edgelist
G_reply = nx.from_pandas_edgelist(sotu_replies,
                                  source='user-screen_name',
                                  target='in_reply_to_screen_name',
                                  create_using=nx.DiGraph())
# Print the number of nodes
print('Nodes in reply network:', len(G_reply.nodes()))
# Print the number of edges
print('Edges in reply network:', len(G_reply.edges()))

'''Visualizing reply network'''
# Create random layout positions
pos = nx.random_layout(G_reply)
# Create size list
sizes = [x[1] for x in G_reply.degree()]
# Draw the network
nx.draw_networkx(G_reply, pos, with_labels = False,
                 node_size =sizes,
                 width =0.1, alpha=0.7,
                 arrowsize =2, linewidths=0)
# Turn axis off and show
plt.axis('off')
plt.show()

'''Centrality: Centrality is a measure of importance of a node to a network.
Centrality:
            - Measures of importance of a node in a network
            - Several different ideas of "importance
1. Degree Centrality = Number of edges that are connected to node
    In-degree - edge going into node
    Out-degree - edge going out of a node
    
2. Betweenness Centrality
 - How many shortest paths between two nodes pass through this node
  '''


'''In-degree centrality'''
column_names = ['screen_name', 'degree_centrality']
# Generate in-degree centrality for retweets
rt_centrality = nx.in_degree_centrality(G_rt)
# Generate in-degree centrality for replies
reply_centrality = nx.in_degree_centrality(G_reply)
# Store centralities in DataFrame
rt = pd.DataFrame(list(rt_centrality.items()), columns = column_names)
reply = pd.DataFrame(list(reply_centrality.items()), columns = column_names)
# Print first five results in descending order of centrality
print('In-degree Centrality for retweets:')
print(rt.sort_values('degree_centrality', ascending = False).head())
# Print first five results in descending order of centrality
print('In-degree Centrality for replies:')
print(reply.sort_values('degree_centrality', ascending = False).head())

'''Betweenness Centrality'''
column_names = ['screen_name', 'betweenness_centrality']
# Generate betweenness centrality for retweets
rt_centrality = nx.betweenness_centrality(G_rt)
# Generate betweenness centrality for replies
reply_centrality = nx.betweenness_centrality(G_reply)
# Store centralities in data frames
rt = pd.DataFrame(list(rt_centrality.items()), columns = column_names)
reply = pd.DataFrame(list(reply_centrality.items()), columns = column_names)
# Print first five results in descending order of centrality
print('Betweenness Centrality for retweets:')
print(rt.sort_values('betweenness_centrality', ascending = False).head())
# Print first five results in descending order of centrality
print('Betweenness Centrality for replies:')
print(reply.sort_values('betweenness_centrality', ascending = False).head())

'''Ratio 
"The Ratio," as it is called, is calculated by taking the number of replies and dividing it by the number of retweets.'''
column_names = ['screen_name', 'degree']
# Calculate in-degrees and store in DataFrame
degree_rt = pd.DataFrame(list(G_rt.in_degree()), columns = column_names)
degree_reply = pd.DataFrame(list(G_reply.in_degree()), columns = column_names)
# Merge the two DataFrames on screen name
ratio = degree_rt.merge(degree_reply, on = 'screen_name', suffixes = ('_rt', '_reply'))
# Calculate the ratio
ratio['ratio'] = ratio['degree_reply'] / ratio['degree_rt']
# Exclude any tweets with less than 5 retweets
ratio = ratio[ratio['degree_rt'] >= 5]
# Print out first five with highest ratio
print('Ratio:')
print(ratio.sort_values('ratio', ascending = False).head())

