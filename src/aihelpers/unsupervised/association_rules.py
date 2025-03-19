import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mlxtend.frequent_patterns import apriori, association_rules

def rules_init(data_ohe_df, min_support, max_len=None, use_colnames=True, metric="confidence", min_threshold=0.8):
    frequent_items = apriori(data_ohe_df, 
                             min_support=min_support, 
                             use_colnames=use_colnames, 
                             max_len=max_len)
    frequent_items.sort_values(by="support", ascending=False, inplace=True)

    if frequent_items.shape[0] > 0:
        association_rules_ = association_rules(frequent_items, 
                                            metric=metric, 
                                            min_threshold=min_threshold)
        association_rules_.sort_values(by="lift", ascending=False, inplace=True)
    else:
        association_rules_ = None

    return frequent_items, association_rules_

def min_support_analysis(data_ohe_df, 
                         log10_support=np.linspace(-2, 0, num=15, endpoint=False),
                         max_len=None, 
                         use_colnames=True, 
                         metric="confidence", 
                         min_threshold=0.8,
                         figsize=(7, 3)):
    """
    Retourne tuple (support, frequent_items, rules) ou chaque element est une liste
    """
    frequent_items = []
    rules = []
    
    # echelle logarithmique pour support; plus de precision avant point d'inflexion
    # remettre echelle lineaire pour traitement
    supports = np.power(10, log10_support)
    
    stats = pd.DataFrame(columns=["min_support", "frequent_items_count", "rules_count"])
    for s in supports:
        frequent_items_, rules_ = rules_init(data_ohe_df, 
                                           s, 
                                           max_len=max_len, 
                                           use_colnames=use_colnames, 
                                           metric=metric, 
                                           min_threshold=min_threshold)
        
        data = pd.DataFrame({"min_support": s, 
                            "frequent_items_count": frequent_items_.shape[0],
                            "rules_count": rules_.shape[0] if not rules_ is None else 0},
                            index=[0])
        stats = pd.concat([stats, data], axis=0, ignore_index=True)
        frequent_items.append(frequent_items_)
        rules.append(rules_)

    plt.figure(figsize=figsize)
    plt.plot(stats.min_support, stats.frequent_items_count, marker=".")
    plt.grid(True)
    plt.xlabel("min_support")
    plt.ylabel("# items frequent")
    plt.show()
    print("Support vs # items frequent/règles")
    display(stats)

    return supports, frequent_items, rules
