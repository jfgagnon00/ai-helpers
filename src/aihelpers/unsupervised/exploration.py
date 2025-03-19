import math
import matplotlib.pyplot as plt
import seaborn as sns

from pandas.api.types import is_integer_dtype, is_numeric_dtype


def pretty_describe(dataframe, decimal=2):
    """
    Utilitaire pour reformatter DataFrame.describe()
    """
    desc = dataframe.describe().round(decimal)
    desc = desc.reindex(["min",
                         "max",
                         "25%",
                         "50%",
                         "75%",
                         "mean",
                         "std"])
    desc = desc.rename(index={"25%": "Q1",
                              "50%": "Q2",
                              "75%": "Q3",
                              "mean": "$\\mu$",
                              "std":  "$\\sigma$"})
    return desc

def show_na(dataframe):
    na_rows = dataframe.isna().any(axis=1)
    na_ = dataframe[na_rows]
    
    na_ratio = na_.shape[0] / dataframe.shape[0]
    na_ratio = round(na_ratio * 100, 1)
    print(f"Valeur manquante {na_.shape[0]} ({na_ratio}%)")
    
    if na_.shape[0] > 0:
        display(na_)

    return na_.index

def show_duplicates(dataframe, show_sorted=True, show_max=None):
    """
    Affiche tous les doublons pour fin d'analyse mais et retourne 
    les index des doublons qui devraient etre enleves (pandans.DataFrame.duplicated(keep="First"))
    """
    dup = dataframe.duplicated()
    dup_ = dataframe[dup]

    dup_ratio = dup_.shape[0] / dataframe.shape[0]
    dup_ratio = round(dup_ratio * 100, 1)
    print(f"Doublons {dup_.shape[0]} ({dup_ratio}%)")

    if dup_.shape[0] > 0:
        all_ = dataframe.duplicated(keep=False)

        dataframe = dataframe[all_]

        if show_sorted:
            by = dataframe.columns.to_list()
            dataframe = dataframe.sort_values(by=by, axis=0)

        print()

        if not show_max is None:
            dataframe = dataframe[:show_max]
            print(f"Les {show_max} premiers doublons (index garde le premier de chaque doublon)")
        else:
            print("Tous les doublons (index garde le premier de chaque doublon)")

        display(dataframe)

    return dup_.index

def pretty_types(dataframe, transpose=True):
    types_ = dataframe.dtypes.to_frame()
    types_.columns = ["Type"]
    return types_

def show_types(dataframe, transpose=True):
    types_ = pretty_types(dataframe)
    print("Types")
    display(types_.T if transpose else types_)

def show_distributions(data, num_cols=5, figsize=(12, 10)):
    num_rows = math.ceil(data.shape[1] / num_cols)
    
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)
    
    for col, ax in zip(data, axes.flatten()):
        if is_integer_dtype(data[col]) or not is_numeric_dtype(data[col]):
            counts = data[col].value_counts()
            sns.barplot(x=counts.index, y=counts.values, ax=ax)
        else:
            sns.histplot(x=col, data=data, ax=ax)
    
    plt.tight_layout()
    plt.show()

def show_outliers_iqr(data, 
                      eta=1.5, 
                      show_outliers_values=False, 
                      boxlists=None, 
                      figsize=(8, 6.5), 
                      show_outliers_cols=True,
                      transpose_outliers_cols=True):
    """
    threshold ~10-15%
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    delta = eta * (q3 - q1)
    
    low = data < (q1 - delta)
    low_rows = low.any(axis=1)

    high = data > (q3 + delta)
    high_rows = high.any(axis=1)
                
    outliers_rows = data[low_rows | high_rows]    
    outliers_ratio_rows = outliers_rows.shape[0] / data.shape[0]
    outliers_ratio_rows = round(outliers_ratio_rows * 100, 1)
    
    outliers_cols = (low | high).sum(axis=0)
    outliers_cols.name = "Count"
    outliers_cols = outliers_cols.to_frame()
    outliers_cols["%"] = round(outliers_cols.Count / data.shape[0] * 100, 1)
    
    print(f"IQR outliers par variable, eta: {eta}")
    if show_outliers_cols:
        display(outliers_cols.T if transpose_outliers_cols else outliers_cols)
        print()
    
    print(f"IQR outliers {outliers_rows.shape[0]} ({outliers_ratio_rows}%), eta: {eta}")    
    if outliers_rows.shape[0] > 0 and show_outliers_values:
        outliers_values = low | high
        outliers_values_rows = outliers_values.loc[outliers_rows.index]
        outliers_values_bg = np.where(outliers_values_rows, "background-color:lightsteelblue", "")
        outliers_values_styler = outliers_rows.style.apply(lambda _: outliers_values_bg, axis=None)
        display(outliers_values_styler)
    else:
        print()
        
    if not boxlists is None: 
        print(f"Outliers boxplots, eta: {eta}")
        n = len(boxlists)
        h_ratios = [len(boxes) for boxes in boxlists]
        fig, axes = plt.subplots(n, figsize=figsize, height_ratios=h_ratios)
        if n == 1:
            sns.boxplot(data[boxlists[0]], orient="h", whis=eta, ax=axes)
        else:
            for boxes, ax in zip(boxlists, axes.flatten()):
                sns.boxplot(data[boxes], orient="h", whis=eta, ax=ax)
        plt.tight_layout()
        plt.show()
        
    return outliers_rows.index

def show_correlation(data, 
                     method='pearson', 
                     corner=True, 
                     figsize=(6, 3), 
                     pairplot=False, 
                     pairplot_figsize=(8, 6),
                     corr_threshold=0):
    corr_ = data.corr(method=method)
    
    plt.figure(figsize=figsize)
    sns.heatmap(corr_, annot=True, linewidths=0.01, fmt=".2f", ax=plt.gca())
    plt.title(f"Corrélation - {method}")
    plt.show()
    
    # mac a quelques problemes avec le temps d'execution du pairplot
    # mettre cette affichage optionel
    if pairplot:
        g = sns.pairplot(data, corner=corner)
        g.fig.set_size_inches(*pairplot_figsize)
        plt.suptitle("Pair plot")
        plt.show()

    if corr_threshold > 0:
        # affiche variables qui ont correlation > que corr_threshold
        corr_ = pd.DataFrame(corr_, index=data.columns, columns=data.columns)
        corr_greater = np.abs(corr_) > corr_threshold
        corr_vars = set()

        for x in corr_.columns:
            for y in corr_.columns:
                if x != y and corr_greater.loc[x, y]:
                    diag_up = (y, x)
                    if diag_up in corr_vars:
                        continue
                    corr_vars.add((x, y))

        print(f"Paires de variables avec corrélaion > |{corr_threshold}|: {len(corr_vars)}")
        display(corr_vars)

        corr_vars2 = set()
        for x, y in corr_vars:
            corr_vars2.add(x)
            corr_vars2.add(y)
        print("Liste précédante 'fusionnée':", len(corr_vars2))
        display(corr_vars2)
