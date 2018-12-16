# coding: utf-8


import numpy as np


HEADERS = ['Onion', 'Potatoe', 'Burger', 'Milk', 'Beer']
ONION    = 0
POTATOE  = 1
BURGER   = 2
MILK     = 3
BEER     = 4

database = [
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 1, 1],
    [1, 1, 0, 1, 0],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 1]
]


def support(database, items):
    """
    The proportion of transactions containing items 'items'.
    """
    nb = 0.0
    for transaction in database:
        found = sum([1 for item in items if transaction[item] == 1]) == len(items)
        if found: nb += 1
    return nb / len(database)

def confidence(database, items1, items2):
    """
    P(Y|X): the probability of finding Y in a transaction that already contains X
    """
    supp_items2 = support(database, items2)
    supp_both   = support(database, items1 + items2)
    return (supp_both) / float(supp_items2)

def lift(database, items1, items2):
    supp_items1 = support(database, items1)
    supp_items2 = support(database, items2)
    supp_both   = support(database, items1 + items2)
    return float(supp_both) / (supp_items1 * supp_items2)

def conviction(database, items1, items2):
    conf = confidence(database, items1, items2)
    supp_items1 = support(database, items1)
    return (1 - supp_items1) / float(1 - conf)

def apriori(database):
    pass

if __name__ == '__main__':
    print('[*] running')
    supp = support(database, [ONION])
    print(f'[*] support "Onion": {supp}')
    supp = support(database, [ONION, POTATOE])
    print(f'[*] support "(Onion, Potatoe)": {supp}')
    conf = confidence(database, [BURGER], [ONION, POTATOE])
    print(f'[*] confidence "(Onion, Potatoe) => (Burger)": {conf}')
    lif  = lift(database, [BURGER], [ONION, POTATOE])
    print(f'[*] lift "(Onion, Potatoe) => (Burger)": {lif}')
    conv = conviction(database, [BURGER], [ONION, POTATOE])
    print(f'[*] conviction "(Onion, Potatoe) => (Burger)": {conv}')