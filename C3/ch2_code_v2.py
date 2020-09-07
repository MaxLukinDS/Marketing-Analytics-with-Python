

import pandas as pd
import numpy as np

'''A library wants to get members to read more and has decided to use market basket analysis to figure out how. 
They approach you to do the analysis and ask that you use the five most highly-rated books from the goodbooks-10k dataset,
 which was introduced in the video. You are given the data in one-hot encoded format in a pandas DataFrame called books.'''



'''Metrics of items assosicaiton and rules filtration:
- Support
- Confidence
- Lift
- Leverage 
- Conviction
- Zhang's metric'''

'''Recommending books with support
The support metric measures the share of transactions that contain an itemset.

Formula: Support(X) = Frequency (X) / N
Formula: Support (X -> Y) = Frequency (X & Y) / N'''

# Compute the support for {Hunger, Potter}
supportHP = np.logical_and(books['Hunger'], books['Potter']).mean()
# Compute the support for {Hunger, Twilight}
supportHT = np.logical_and(books['Hunger'], books['Twilight']).mean()
# Compute the support for {Potter, Twilight}
supportPT = np.logical_and(books['Potter'], books['Twilight']).mean()
# Print support values
print("Hunger Games and Harry Potter: %.2f" % supportHP)
print("Hunger Games and Twilight: %.2f" % supportHT)
print("Harry Potter and Twilight: %.2f" % supportPT)

'''# Refining support with confidence'''
'''The confidence metric: 
Can improve over support with additional metrics. Adding confidence provides a more complete picture

Formula: Support(X & Y) / Support(X) '''

# You'll compute it for both {Potter} → {Twilight} and {Twilight} → {Potter}.
# Compute the support of {Potter, Twilight}
supportPT = np.logical_and(books['Potter'], books['Twilight']).mean()
# Compute the support of {Potter}
supportP = books['Potter'].mean()
# Compute the support of {Twilight}
supportT = books['Twilight'].mean()
# Compute the confidence of {Potter} → {Twilight} and {Twilight} → {Potter}
confidencePT = supportPT / supportP
confidenceTP = supportPT / supportT
# Print results
print('{0:.2f}, {1:.2f}'.format(confidencePT, confidenceTP))

'''Further refinement with lift'''
'''Lift provides another metric for evaluating the relationship between items.
Numerator: Proportion of transactions that contain X and Y.
Denominator: Proportion if X and Y assigned randomly and independently.

Formula: Support(X & Y) / [Support(X) * Support(Y)] '''

# Compute the support of {Potter, Twilight}
supportPT = np.logical_and(books['Potter'], books['Twilight']).mean()
# Compute the support of {Potter}
supportP = books['Potter'].mean()
# Compute the support of {Twilight}
supportT = books['Twilight'].mean()
# Compute the lift of {Potter} → {Twilight}
lift = supportPT / (supportP * supportT)
# Print lift
print("Lift: %.2f" % lift)

'''The leverage metric
Leverage is similar to lift, but easier to interpret. Leverage lies in -1 and +1 range. Lift ranges from 0 to infinity

Formula: Leverage(X → Y ) = Support(X&Y ) − Support(X) * Support(Y )'''

# Computing leverage
# Compute support for {Twilight} and {Harry Potter}
supportTP = np.logical_and(books['Twilight'], books['Potter']).mean()
# Compute support for {Twilight}
supportT = books['Twilight'].mean()
# Compute support for {Harry Potter}
supportP = books['Potter'].mean()

# Compute and print leverage
leverage = supportTP - supportP * supportT
print(leverage)



'''Conviction 
Conviction is also built using support. More complicated and less intuitive than leverage.

Formula: Conviction (X -> Y) = Support(X) * Support(avg Y) / Support(X & avg Y)'''

# Compute the support for {Potter} and {Hunger}
supportPH = np.logical_and(books['Potter'], books['Hunger']).mean()
# Compute the support for {Potter}
supportP = books['Potter'].mean()
# Compute the support for NOT {Hunger}
supportnH = 1.0 - books['Hunger'].mean()
# Compute the support for {Potter} and NOT {Hunger}
supportPnH = supportP - supportPH
# Compute and print conviction for {Potter} -> {Hunger}
conviction = supportP * supportnH / supportPnH
print("Conviction: %.2f" % conviction)

'''Computing conviction with a function'''
def conviction(antecedent, consequent):
    # Compute support for antecedent AND consequent
    supportAC = np.logical_and(antecedent, consequent).mean()
    # Compute support for antecedent
    supportA = antecedent.mean()
    # Compute support for NOT consequent
    supportnC = 1.0 - consequent.mean()
    # Compute support for antecedent and NOT consequent
    supportAnC = supportA - supportAC
    # Return conviction
    return supportA * supportnC / supportAnC

'''Promoting ebooks with conviction'''
# Compute conviction for {Twilight} → {Potter} and {Potter} → {Twilight}
convictionTP = conviction(twilight, potter)
convictionPT = conviction(potter, twilight)
# Compute conviction for {Twilight} → {Hunger} and {Hunger} → {Twilight}
convictionTH = conviction(twilight, hunger)
convictionHT = conviction(hunger, twilight)
# Compute conviction for {Potter} → {Hunger} and {Hunger} → {Potter}
convictionPH = conviction(potter, hunger)
convictionHP = conviction(hunger, potter)

# Print results
print('Harry Potter -> Twilight: ', convictionPT)
print('Twilight -> Potter: ', convictionTP)

''' Using dissociation '''

''' Zhang's metric
1. Introduced by Zhang (2000)
Takes values between -1 and +1
Value of +1 indicates perfect association
Value of -1 indicates perfect dissociation
2. Comprehensive and interpretable
3. Constructed using support 

Formula (1):
Zhang(A → B) =  [ Confidence(A → B) − Confidence( avg A → B) ] / [ Max(Confidence(A → B),Confidence( avg A → B)) ] ,
    where Confidence = Support(A & B) / Support(A)
    
Formula (2) using Support:
Zhang(A → B) = [ Support(A & B) − Support(A)Support(B) ] / 
        [ Max[Support(AB)(1 − Support(A)), Support(A)(Support(B) − Support(AB)] ]'''

'''Computing association and dissociation'''
# Compute the support of {Twilight} and the support of {Potter}
supportT = books['Twilight'].mean()
supportP = books['Potter'].mean()
# Compute the support of {Twilight, Potter}
supportTP = np.logical_and(books['Twilight'], books['Potter']).mean()
# Complete the expressions for the numerator and denominator
numerator = supportTP - supportT*supportP
denominator = max(supportTP*(1-supportT), supportT*(supportP-supportTP))
# Compute Zhang's metric for {Twilight} → {Potter}
zhang = numerator / denominator
print(zhang)

'''Define a function to compute Zhang's metric'''
def zhang(antecedent, consequent):
    # Compute the support of each book
    supportA = antecedent.mean()
    supportC = consequent.mean()
    # Compute the support of both books
    supportAC = np.logical_and(antecedent, consequent).mean()
    # Complete the expressions for the numerator and denominator
    numerator = supportAC - supportA*supportC
    denominator = max(supportAC*(1-supportA), supportA*(supportC-supportAC))

    # Return Zhang's metric
    return numerator / denominator

'''Applying Zhang's metric'''
# Define an empty list for Zhang's metric
zhangs_metric = []

# Loop over lists in itemsets
for itemset in itemsets:
    # Extract the antecedent and consequent columns
    antecedent = books[itemset[0]]
    consequent = books[itemset[1]]
    # Complete Zhang's metric and append it to the list
    zhangs_metric.append(zhang(antecedent, consequent))
# Print results
rules['zhang'] = zhangs_metric
print(rules)

'''How does filtering work?

Support {Potter} → {Hunger} = 0.001 might be excluded 
Lift {Hobbit} → {Twilight} = 0.85 might be excluded 
'''

'''Filtering with support and conviction'''
# Preview the rules DataFrame using the .head() method
print(rules.head())
# Select the subset of rules with antecedent support greater than 0.05
rules = rules[rules['antecedent support'] > 0.05]
# Select the subset of rules with a consequent support greater than 0.01
rules = rules[rules['consequent support'] > 0.01]
# Select the subset of rules with a conviction greater than 1.01
rules = rules[rules['conviction'] > 1.01]
# Print remaining rules
print(rules)

'''Using multi-metric filtering to cross-promote books'''
# Set the lift threshold to 1.5
rules = rules[rules['lift'] > 1.5]
# Set the conviction threshold to 1.0
rules = rules[rules['conviction'] > 1.0]
# Set the threshold for Zhang's rule to 0.65
rules = rules[rules['zhang'] > 0.65]
# Print rule
print(rules[['antecedents','consequents']])



