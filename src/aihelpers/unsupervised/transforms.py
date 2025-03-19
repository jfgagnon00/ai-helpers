import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import seaborn as sns
import warnings

from collections import namedtuple


def transform_boxcox(data, columns=None):
    """
    Retourne tuple (transforms, lambdas)
    """
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    lambdas_ = []
    for c in columns:
        doi = data[c]
        doi_min = doi.min()
        if doi_min <= 0:
            doi = doi - doi_min + 1e-6
        
        result[c], lambda_ = sp.stats.boxcox(doi)
        lambdas_.append(lambda_)
        
    return result, lambdas_

def transform_log(data, columns=None):
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    for c in columns:
        doi = data[c]
        doi_min = doi.min()
        if doi_min < 1:
            doi = doi - doi_min + 1e-6
        
        result[c] = np.log(doi + 1)
        
    return result

def transform_sqrt(data, columns=None):
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    for c in columns:
        doi = data[c]
        doi_min = doi.min()
        if doi_min < 0:
            doi = doi - doi_min
        
        result[c] = np.sqrt(doi)
        
    return result

def transform_cbrt(data, columns=None):
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    for c in columns:
        doi = data[c]
        result[c] = np.cbrt(doi)
        
    return result

def show_transforms(data, columns=None, figsize=(10, 10)):
    """
    Applique et afiche diverses transformes et les retourne 
    sous forme de namedtuple
    Attention, boxcox est un tuple (DataFrame, boxcox lambda)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        transforms_names = ["log", 
                            "boxcox",
                            "cbrt"]
            
        Transforms = namedtuple("Transforms", transforms_names)
        transforms = Transforms(
            transform_log(data, columns=columns),
            transform_boxcox(data, columns=columns),
            transform_cbrt(data, columns=columns))

    if columns is None:
        columns = data.columns

    _, axes = plt.subplots(len(columns), len(transforms_names) + 1, figsize=figsize, squeeze=False)

    for col_index, col_name in enumerate(columns):
        axes[col_index, 0].set_ylabel(col_name)

        # normal dist
        g = sns.histplot(data[col_name].to_numpy(), ax=axes[col_index, 0])
        g.set(xlabel=None)

        # log
        new_dist = transforms.log[col_name]
        g = sns.histplot(new_dist.to_numpy(), ax=axes[col_index, 1])
        g.set(xlabel=None, ylabel=None)

        # cox box 
        new_dist = transforms.boxcox[0][col_name]
        g = sns.histplot(new_dist, ax=axes[col_index, 2])
        g.set(xlabel=None, ylabel=None)

        # cubic root
        new_dist = transforms.cbrt[col_name]
        g = sns.histplot(new_dist, ax=axes[col_index, 3])
        g.set(xlabel=None, ylabel=None)

    axes[0, 0].set_title("Dist. Originale")
    axes[0, 1].set_title("Log")
    axes[0, 2].set_title("Boxcox")
    axes[0, 3].set_title("Cubic Root")

    plt.tight_layout()
    plt.show()

    return transforms
