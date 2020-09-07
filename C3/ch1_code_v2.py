
import pandas as pd
import numpy as np

# Load transactions from pandas
groceries = pd.read_csv('small_grocery_store.csv')

# Preparing data for market basket analysis
# Split transaction strings into lists
transactions = groceries['Transaction'].apply(lambda t: t.split(','))
# Convert DataFrame column into list of strings
transactions = list(transactions)
# Print the list of transactions
print(transactions)

# Generating association rules
'''Difficulty of selecting the rules 
Finding useful rules is difficult. 
- Set of all possible rules is large 
- Most rules are not useful
- Most discard most rules '''
# Import permutations from the itertools module
from itertools import permutations
# Define the set of groceries
flattened = [i for t in transactions for i in t]
groceries = list(set(flattened))
# Generate all possible rules
rules = list(permutations(groceries, 2))
# Print the set of rules
print(rules)
# Print the number of rules
print(len(rules))

'''Metrics and pruning:
A metric is a measure of performance for rules. 
Pruning is the use of metrics to discard rules.'''

'''The support metric:
    measures the share of transactions that contain an itemset.'''
# One-hot encoding transaction data
# Import the transaction encoder function from mlxtend
from mlxtend.preprocessing import TransactionEncoder
# Instantiate transaction encoder and identify unique items
encoder = TransactionEncoder().fit(transactions)
# One-hot encode transactions
onehot = encoder.transform(transactions)
# Convert one-hot encoded data to DataFrame
onehot = pd.DataFrame(onehot, columns = encoder.columns_)
# Print the one-hot encoded transaction dataset
print(onehot)

# Computing the support metric
# Print the support
print(onehot.mean())

# to check whether the rule {jam} → {bread} has a support of over 0.05
# Add a jam+bread column to the DataFrame onehot
onehot['jam+bread'] = np.logical_and(onehot['jam'], onehot['bread'])
# Compute the support
support = onehot.mean()
# Print the support values
print(support)


