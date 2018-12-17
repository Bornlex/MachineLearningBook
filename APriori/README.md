# Apriori algorithm

## Concepts

### Support

The support of an itemset X, supp(X), is the proportion of transaction in the database in which the item X appears.

**supp(X)** = (number of transaction in which X appears) / (total number of transactions)

### Confidence

It signifies the likelihood of item Y being purchased when item X is purchased.

**conf(X -> Y)** = supp(X U Y) / supp(X)

### Lift

This signifies the likelihood of the itemset Y being purchased when item X is purchased while taking into account the popularity of Y.

**lift(X -> Y)** = supp(X U Y) / (supp(X) * supp(Y))

### Conviction

**conv(X -> Y)** = (1 - supp(X)) / (1 - conf(X -> Y))

The conviction value of 1.32 means that the rule {onion,potato}=>{burger} would be incorrect 32% more often if the association between X and Y was an accidental chance.

## A priori

Apriori algorithm is a classical algorithm in data mining. It is used for mining frequent itemsets and association rules in a database.

It is basically mining itemsets which support is above a given threshold.

There are 2 main rules to improve the efficiency of the algorithm:
1. all subsets of a frequent itemset must be frequent
2. for any infrequent itemset, all super itemsets must be infrequent too (so it is useless to test them)