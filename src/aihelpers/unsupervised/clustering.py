import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from collections import namedtuple
from matplotlib.ticker import MaxNLocator

from fanalysis.mca import MCA
from fanalysis.pca import PCA as fa_PCA

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.decomposition import PCA as sk_PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, \
                            silhouette_samples, \
                            davies_bouldin_score

def pca_init(std_data, n_components):
    # pour fin de comparaison, choisir au runtime entre fanalysis et scklearn
    if False:
        acp = fa_PCA(std_unit=False, 
                  n_components=n_components,
                  row_labels=std_data.index,
                  col_labels=std_data.columns)
        acp.fit(std_data.to_numpy())
    else:
        acp = sk_PCA(n_components=n_components, 
                     svd_solver="full")

        # fanalysis adapters
        acp.row_coord_ = acp.fit_transform(std_data)
        acp.row_labels_ = std_data.index
        acp.col_labels_ = std_data.columns
        acp.eig_ = [acp.explained_variance_,
                    acp.explained_variance_ratio_ * 100,
                    np.cumsum(acp.explained_variance_ratio_ * 100)]

    def show2d(coords, coords_labels, alpha):
        plt.scatter(coords[:, 0], coords[:, 1])

        for xy, text in zip(coords, coords_labels):
            text_ = plt.text(xy[0], xy[1], text)
            text_.set_alpha(alpha)
                
        plt.grid(True)
        
    def show(x, y, text_alpha=0.33, figsize=(5, 4)):
        plt.figure(figsize=figsize)
        show2d(acp.row_coord_[:, [x - 1, y - 1]], \
               acp.row_labels_, \
               text_alpha)
        plt.show()

    # override mapping_row
    acp.mapping_row = show
    
    return acp

def pca_analysis(std_data, figsize=(4, 2.5), print_eig_values=True, print_variance=True):
    """
    Le threshold est ~60% sur cumul var. expliquee
    """
    acp = pca_init(std_data, None)
    
    saporta = 1 + 2 * math.sqrt((std_data.shape[1] - 1) / (std_data.shape[0] - 1))

    eig_vals = acp.eig_[0]
    eig_th0 = eig_vals[eig_vals > 1]
    eig_th1 = eig_vals[eig_vals > saporta]

    if print_eig_values:
        print("Valeurs propres:")
        print(acp.eig_[0].round(4))
        print()
        print("Valeurs propres > 1:")
        print(eig_th0.round(4))
        print()
        print(f"Valeurs propres > {round(saporta, 4)} (saporta):")
        print(eig_th1.round(4))
        print()

    if print_variance:
        print("Variance expliquee %:")
        print(acp.eig_[1].round(1))
        print()
        print("Variance expliquee cumul. %:")
        print(acp.eig_[2].round(1))
        print()

    num_eigval = len(acp.eig_[0])
    
    plt.figure(figsize=figsize)
    plt.plot(range(1, num_eigval + 1), acp.eig_[0], marker=".")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("# axe factoriel")
    plt.ylabel("Valeur propre")
    plt.show()
    
def acm_init(data, n_components):
    acm = MCA(n_components=n_components,
              row_labels=data.index,
              var_labels=data.columns)
    acm.fit(data.to_numpy())
    return acm

def acm_analysis(data, figsize=(4, 2.5)):
    """
    Le threshold est ~60% sur cumul var. expliquee
    """
    acm = acm_init(data, None)
    
    threshold = 1 / data.shape[1]
    eig_vals = acm.eig_[0]
    eig_th = eig_vals[eig_vals > threshold]

    print("Valeurs propres:")
    print(acm.eig_[0].round(4))
    print()
    print(f"Valeurs propres > {round(threshold, 4)} (1 / p):")
    print(eig_th.round(4))
    print()
    print("Variance expliquee %:")
    print(acm.eig_[1].round(1))
    print()
    print("Variance expliquee cumul. %:")
    print(acm.eig_[2].round(1))
    print()

    num_eigval = len(acm.eig_[0])

    plt.figure(figsize=figsize)
    plt.plot(range(1, num_eigval + 1), acm.eig_[0], marker=".")
    plt.grid(True)
    plt.xlabel("# axe factoriel")
    plt.ylabel("Valeur propre")
    plt.show()

def kmeans_init(coords, n_clusters, n_init=20, max_iter=300, use_mini_batch=False):
    if use_mini_batch:
        clstr = MiniBatchKMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    else:
        clstr = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    clstr.fit(coords)
    return clstr
    
def kmeans_analysis(coords, 
                    clusters_range=range(2, 15), 
                    n_init=20, 
                    max_iter=300,
                    use_mini_batch=False,
                    figsize=(5, 3)):
    inertias = []
    sil_scores = []
    
    for k in clusters_range:
        clstr = kmeans_init(coords, k, n_init, max_iter, use_mini_batch)
        inertias.append(clstr.inertia_)
        sil_scores.append( silhouette_score(coords, clstr.labels_) )
    
    _, ax1 = plt.subplots(figsize=figsize)
    
    ax1.plot(clusters_range, inertias, label="wss", color="green", marker=".")
    ax1.set_xlabel("# clusters")
    ax1.set_ylabel("wss")
    ax1.grid(True)
    
    ax2 = plt.gca().twinx()
    ax2.plot(clusters_range, sil_scores, label="silhouette score", color="blue", marker=".")
    ax2.set_ylabel("silhouette score")

    g1, gl1 = ax1.get_legend_handles_labels()
    g2, gl2 = ax2.get_legend_handles_labels()
    plt.legend(g1 + g2, gl1 + gl2)

    plt.show()
    
def cah_init(coords, n_clusters):
    cah = AgglomerativeClustering(n_clusters=n_clusters, 
                                  metric="euclidean", 
                                  linkage='ward')
    cah.fit(coords)
    return cah

def cah_analysis(coords, method="ward", metric="euclidean", figsize=(12, 3.5), num_clusters=20):
    """
    Le threshold est a peu pres a la moitie de la hauteur
    """
    linkage_ = linkage(coords, method=method, metric=metric)

    plt.figure(figsize=figsize)
    plt.subplot(121)
    dendrogram(linkage_)
    plt.title("Dendogramme")
        
    cluster_inertias = linkage_[-num_clusters:, 2]
    cluster_inertias = cluster_inertias[::-1]
    
    plt.subplot(122)
    plt.step(range(2, len(cluster_inertias) + 2), cluster_inertias)
    plt.xlabel("# clusters")
    plt.ylabel("Inertie")
    plt.grid()
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    
    plt.show()
    
def dbscan_init(coords, eps, min_samples):
    dbs = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbs.fit(coords)
    return dbs

def dbscan_eps_analysis(coords, figsize=(5, 3), ylim=None):
    nn = NearestNeighbors(n_neighbors=2)
    distances, _ = nn.fit(coords).kneighbors(coords)
    distances = np.sort(distances, axis=0)

    plt.figure(figsize=figsize)
    plt.plot(distances[:,1], marker=".")
    if not ylim is None:
        plt.ylim(ylim)
    plt.ylabel("Distance")
    plt.xlabel("Individu")
    plt.grid(True)
    plt.show()

def dbscan_parameters_analysis(coords, eps_range, min_samples_range):
    score_max = -1e9
    eps_ = 0
    min_samples_ = 0
    n_clusters_ = 0
    with_ouliers_ = False
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbs = dbscan_init(coords, eps, min_samples)

            clusters = set(dbs.labels_)
            with_ouliers = -1 in clusters
            n_clusters = len(clusters) - (1 if with_ouliers else 0)

            # si dbscan ne donne que 1 seul cluster (-1, outliers)
            # bail out            
            if n_clusters < 2:
                continue
            
            score = silhouette_score(coords, dbs.labels_)
            if score > score_max:
                score_max = score
                eps_ = eps
                min_samples_ = min_samples
                n_clusters_ = n_clusters
                with_ouliers_ = with_ouliers
                    
    print("DBSCAN optimal parameters")
    print("eps:", eps_)
    print("min_samples:", min_samples_)
    print("silhouette score:", round(score_max, 4))
    print("# clusters:", n_clusters_, "+ ouliers" if with_ouliers_ else "(no outliers)")
                
    return eps_, min_samples_

def dbscan_outliers_analysis(coords, eps_range, min_samples, figsize=(5, 3)):
    outliers_ratio = []
    for eps in eps_range:
        dbs = dbscan_init(coords, eps, min_samples)
        outliers = dbs.labels_[dbs.labels_ == -1]
        ratio = len(outliers) / len(dbs.labels_)
        outliers_ratio.append(ratio)
        
    plt.figure(figsize=figsize)
    plt.plot(eps_range, outliers_ratio, marker=".")
    plt.xlabel("dbscan epsilon")
    plt.ylabel("outliers ratio")
    plt.grid(True)
    plt.show()

def clusters_analysis(coords, labels, original_data=None):
    score = davies_bouldin_score(coords, labels)
    print("Davies Bouldin score:", round(score, 4))

    print()

    score = silhouette_score(coords, labels)
    print("Silhouette score:", round(score, 4))

    samples = silhouette_samples(coords, labels)
    samples_means = []
    clusters = set(labels)
    for k in clusters:
        labels_k = labels == k
        
        if labels_k.any() > 0:
            sample_mean = samples[labels_k].mean()
            sample_mean = round(sample_mean, 4)
            samples_means.append(sample_mean)
        else:
            samples_means.append(np.nan)
        
    print("Silhouette score par cluster")
    print(samples_means)
    print()
    
    if not original_data is None:
        tss = (original_data.mean() - original_data) ** 2
        tss = tss.sum(axis=0)

        groups = original_data.groupby(labels)

        bss = (original_data.mean() - groups.mean()) ** 2
        bss = bss.multiply(groups.size(), axis=0)
        bss = bss.sum(axis=0)

        r2 = bss / tss
        r2.name = "$R^2$"

        r2.sort_values(ascending=False, inplace=True)

        print("Clusters means")
        display(groups.mean())

        print()

        print("Variance expliquée par les clusters (triée par R2)")
        display(r2.round(3).to_frame().T)

        return r2
    
    return None

def scatter_plot(coords_, figsize=(5, 4), marker_size=None):
    plt.figure(figsize=figsize)
    plt.scatter(coords_[:, 0], coords_[:, 1], s=marker_size)
    plt.grid(True)
    plt.show()

def scatter_multiplot(coords_, 
                      num_cols=5, 
                      figsize=(10, 6), 
                      marker_size=None, 
                      labels=None,
                      show_legend=False):
    num_dimensions = coords_.shape[1]
    num_graphs = num_dimensions - 1
    num_rows = math.ceil(num_graphs / num_cols)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize, squeeze=False)

    if labels is None:
        for coord, ax in zip(range(num_graphs), axes.flatten()):
            x = coords_[:, coord]
            y = coords_[:, coord + 1]
            ax.scatter(x, y, s=marker_size)
            ax.set_title(f"Coords: {coord}, {coord + 1}")
            ax.grid(True)
    else:
        clusters = set(labels)
        n_clusters = len(clusters)
        
        colors_steps = np.arange(0, 1, 1 / n_clusters)
        colors = plt.cm.nipy_spectral(colors_steps)

        for coord, ax in zip(range(num_graphs), axes.flatten()):
            for k in clusters:
                rows = labels == k
                label = "Outliers" if k == -1 else f"Cluster_{k}"

                x = coords_[rows, coord]
                y = coords_[rows, coord + 1]
                ax.scatter(x, y, label=label, color=colors[k], s=marker_size)
                ax.set_title(f"Coords: {coord}, {coord + 1}")
                ax.grid(True)

        if show_legend:
            axes[0, 0].legend()

    plt.tight_layout()
    plt.show()

def show_clusters(coords_, coords_name_, labels, figsize=(5, 4), text_alpha=1, marker_size=None):
    clusters = set(labels)
    n_clusters = len(clusters)
    
    colors_steps = np.arange(0, 1, 1 / n_clusters)
    colors = plt.cm.nipy_spectral(colors_steps)
    
    plt.figure(figsize=figsize)
    for k in clusters:
        cluster = labels == k
        if isinstance(coords_, pd.DataFrame):
            coords = coords_[cluster].to_numpy()
        else:
            coords = coords_[cluster]
        coords_name = coords_name_[cluster]

        label = "Outliers" if k == -1 else f"Cluster_{k}"
        plt.scatter(coords[:, 0], coords[:, 1], label=label, color=colors[k], s=marker_size)
        
        for xy, text in zip(coords, coords_name_):
            text_ = plt.text(xy[0], xy[1], text)
            text_.set_alpha(text_alpha)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1.0))
    plt.show()