# coding: utf-8


import numpy as np


identity = np.identity(5)
HEADERS = ['Onion', 'Potatoe', 'Burger', 'Milk', 'Beer']
ONION    = identity[0]
POTATOE  = identity[1]
BURGER   = identity[2]
MILK     = identity[3]
BEER     = identity[4]

database = np.mat([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1]
])


def support(database, items):
    """
    Proportion of transactions containing items 'items'.
    """
    #import pdb; pdb.set_trace()
    results = items * database.T
    tx_nb   = sum([1 for result in results.T if np.sum(result) >= np.sum(items)])
    return float(tx_nb) / len(database)

def confidence(database, items1, items2):
    """
    P(Y|X): probability of finding Y in a transaction that already contains X.
    """
    supp_items2 = support(database, items2)
    supp_both   = support(database, np.r_[items1, items2])
    return (supp_both) / float(supp_items2)

def lift(database, items1, items2):
    supp_items1 = support(database, items1)
    supp_items2 = support(database, items2)
    supp_both   = support(database, np.r_[items1, items2])
    return float(supp_both) / (supp_items1 * supp_items2)

def conviction(database, items1, items2):
    conf = confidence(database, items1, items2)
    supp_items1 = support(database, items1)
    return (1 - supp_items1) / float(1 - conf)

def arrays_equal(array1, array2):
    for index in range(len(array1)):
        if array1[index] != array2[index]:
            return False
    return True

def transaction_in_matrix(matrix, array):
    for row in matrix:
        if arrays_equal(row, array): return True 
    return False

def get_uniques(database):
    uniques = []
    for transaction in database:
        if not transaction_in_matrix(uniques, transaction) and 2 not in transaction:
            uniques.append(transaction)
    return np.mat(uniques)

def generate(old_generation, first_generation):
    new_generation = None
    for i, item in enumerate(first_generation):
        if new_generation is None:
            new_generation = np.r_[(old_generation + item)[:i], (old_generation + item)[i + 1:]]
        else:
            new_generation = np.r_[new_generation, np.r_[(old_generation + item)[:i], (old_generation + item)[i + 1:]]]
    return get_uniques(new_generation)

def apriori(database, threshold):
    itemset = None
    generation = identity.copy()
    while len(generation) != 0:
        generation = np.array([i for i in generation if support(database, np.mat(i)) >= threshold])
        if len(generation) == 0:
            break
        if itemset is None: itemset = generation
        else              : itemset = np.r_[itemset, generation]
        generation = np.array(generate(generation, identity))
    return itemset

if __name__ == '__main__':
    print('[*] running')
    supp = support(database, np.mat([ONION]))
    print(f'[*] support "Onion": {supp}')
    supp = support(database, np.mat([ONION, POTATOE]))
    print(f'[*] support "(Onion, Potatoe)": {supp}')
    conf = confidence(database, np.mat([BURGER]), np.mat([ONION, POTATOE]))
    print(f'[*] confidence "(Onion, Potatoe) => (Burger)": {conf}')
    lif  = lift(database, np.mat([BURGER]), np.mat([ONION, POTATOE]))
    print(f'[*] lift "(Onion, Potatoe) => (Burger)": {lif}')
    conv = conviction(database, np.mat([BURGER]), np.mat([ONION, POTATOE]))
    print(f'[*] conviction "(Onion, Potatoe) => (Burger)": {conv}')
    threshold = 0.5
    print(f'[*] a priori:')
    print(apriori(database, threshold))