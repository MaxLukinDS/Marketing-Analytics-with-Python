import pandas as pd
import seaborn as sns

'''Здась используется online_retail'''

# Load ratings data.
ratings = pd.read_csv('movielens_movies.csv')

'''
Preparing the data
1. Generate the rules.
Use Apriori algorithm and association rules.
2. Convert antecedents and consequents into strings.
Stored as frozen sets by default in mlxtend.
3. Convert rules into matrix format.
Suitable for use in heatmaps. '''

'''Visualizing itemset support'''
# Compute frequent itemsets using a minimum support of 0.07
frequent_itemsets = apriori(onehot, min_support=0.07, use_colnames=True, max_len=2)
# Compute the association rules
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.0)

# Replace frozen sets with strings
rules['antecedents'] = rules['antecedents'].apply(lambda a: ','.join(list(a)))
rules['consequents'] = rules['consequents'].apply(lambda a: ','.join(list(a)))

# Transform data to matrix format and generate heatmap
pivot = rules.pivot(index='consequents', columns='antecedents', values='support')
sns.heatmap(pivot)

# Format and display plot
plt.yticks(rotation=0)
plt.show()

'''Heatmaps with lift'''
# Transform the DataFrame of rules into a matrix using the lift metric
pivot = rules.pivot(index='consequents', columns='antecedents', values='lift')

# Generate a heatmap with annotations on and the colorbar off
sns.heatmap(pivot, annot=True, cbar=False)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.show()

'''Pruning with scatterplots'''
# Apply the Apriori algorithm with a support value of 0.0075
frequent_itemsets = apriori(onehot, min_support=0.0075, use_colnames=True, max_len=2)
# Generate association rules without performing additional pruning
rules = association_rules(frequent_itemsets, metric='support', min_threshold=0.0)
# Generate scatterplot using support and confidence
sns.scatterplot(x="support", y="confidence", data=rules)
plt.show()

'''Optimality of the support-confidence border'''
# Apply the Apriori algorithm with a support value of 0.0075
frequent_itemsets = apriori(onehot, min_support=0.0075, use_colnames=True, max_len=2)
# Generate association rules without performing additional pruning
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.0)
# Generate scatterplot using support and confidence
sns.scatterplot(x="support", y="confidence", size="lift", data=rules)
plt.show()

'''Using parallel coordinates to visualize rules'''
# Compute the frequent itemsets
frequent_itemsets = apriori(onehot, min_support=0.05, use_colnames=True, max_len = 2)
# Compute rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.50)
# Convert rules into coordinates suitable for use in a parallel coordinates plot
coords = rules_to_coordinates(rules)

# Generate parallel coordinates plot
parallel_coordinates(coords, 'rule')
plt.legend([])
plt.show()

'''Refining a parallel coordinates plot'''
# Import the parallel coordinates plot submodule
from pandas.plotting import parallel_coordinates
# Convert rules into coordinates suitable for use in a parallel coordinates plot
coords = rules_to_coordinates(rules)
# Generate parallel coordinates plot
parallel_coordinates(coords, 'rule', colormap='ocean')
plt.legend([])
plt.show()