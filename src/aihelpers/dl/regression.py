import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.inspection import DecisionBoundaryDisplay
from sys import float_info

from .LogisticRegressionBoundaryAdapter import LogisticRegressionBoundaryAdapter

def ordinary_least_square(x, y):
    """
    Resout l'equation Y=X theta de maniere analytique.

    Args:
        x: 
            tenseur numpy
        
        y: 
            tenseur numpy
    
    Retourne:
        tuple (intercept, coefs)

    Notes:
        - les echantillons sont dans les rangees
        - les variables sont dans les colonnes
        - l'intercept n'est pas suppose etre dans les colonnes
    """
    # np.c_[] est pour concatener colonnes
    # ai mis intercept a la fin - plus naturel pour moi meme si theorie le met au debut
    x_b = np.c_[x, np.ones(x.shape[0])]

    theta = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y

    return theta[-1], theta[:-1].ravel()

def ordinary_gradient_descent(x, y, eta, max_iterations=1000, tolerance=-1, return_y_hats=False):
    """
    Resout l'equation Y=X theta en utilisant la descente 
    de gradient ordinaire.

    Args:
        x: 
            tenseur numpy (assume contenir l'intercept)
        
        y: 
            tenseur numpy

        eta: 
            pas d'apprentissage
        
        max_iterations: 
            maximum d'iterations avant que l'algo termine

        tolerance: 
            Threshold de stabilite pour terminer algo. <= 0 fait en sorte que toutes 
            les iterations sont faites.

        return_y_hats:
            True retourne y_hat a chaque iteration 
            False ne retourne pas y_hat

    Retourne:
        si return_y_hats est True
            tuple (intercep, coefs, MSE a chaque iteration, y_hat a chaque iteration)
        sinon
            tuple (intercep, coefs, MSE a chaque iteration)

        MSE => Mean Square Error

    Notes:
        - les echantillons sont dans les rangees
        - les variables sont dans les colonnes
        - l'intercept n'est pas suppose etre dans les colonnes
    """
    shape_ = x.shape
    shape_len = len(shape_)
    num_samples = shape_[0] if shape_len > 0 else shape_
    num_variables = shape_[1] if shape_len > 1 else 1
    degree = shape_[2] if shape_len > 2 else 1

    # np.c_[] est pour concatener colonnes
    # ai mis intercept a la fin - plus naturel pour moi meme si theorie le met au debut
    x_b = np.c_[x, np.ones(num_samples)]

    # initialiser algo
    errors = []
    last_error = float_info.max
    thetas = np.random.random(size=(num_variables + 1, degree))

    y_hat = x_b @ thetas
    y_hats = None
    if return_y_hats:
        y_hats = []

    # y_hat va etre 2D pour tenir compte de l'intercept
    # ajuster y pour avoir la meme dimension
    if len(y.shape) == 1:
        y = y[:, np.newaxis]

    for i in range(max_iterations):
        # calculer gradient
        gradient = x_b.T @ (y_hat - y) * 2 / num_samples

        # mettre a jour theta
        thetas -= eta * gradient

        y_hat = x_b @ thetas

        # logger pour retour
        if return_y_hats:
            y_hats.append(y_hat.ravel())

        # regarder evolution de l'erreur
        error = np.square(y - y_hat)
        error = np.mean(error)
        errors.append(error)

        delta_error = abs(error - last_error)
        last_error = error
        if delta_error < tolerance:
            # arrete progression erreur est suffisament petite
            # TODO: mettre critere de stabilite avec N delta_error < tolerance
            break

    if return_y_hats:
        return thetas[-1], thetas[:-1].ravel(), errors, y_hats

    return thetas[-1], thetas[:-1].ravel(), errors

def plot_decision_boundaries_2d(classifier, 
                                x, 
                                y_true, 
                                y_pred, 
                                classes, 
                                markers,
                                lut,
                                alpha=0.5, 
                                eps=1,
                                class_of_interest=None,
                                class_probabilities=None,
                                class_probabilities_colors=None,
                                grid_resolution=100, 
                                ax=None,
                                add_legend=True):
    """
    Affiche frontieres de decision. Preter attention a la legende.
    marker == predicted class, couleur == true class
    """
    if isinstance(lut, dict):
        cmap = ListedColormap(lut.values())
    else:
        cmap = ListedColormap(lut)

    # frontieres de decision
    decision_bound_dsp = DecisionBoundaryDisplay.from_estimator(classifier, 
                                                                x, 
                                                                eps=eps,
                                                                response_method="predict",
                                                                grid_resolution=grid_resolution,
                                                                alpha=alpha,
                                                                cmap=cmap,
                                                                ax=ax)
    
    if not class_of_interest is None:
        # probabilites pour class_of_interest
        probas_dsp = DecisionBoundaryDisplay.from_estimator(classifier, 
                                                            x, 
                                                            eps=eps,
                                                            plot_method="contour",
                                                            response_method="predict_proba",
                                                            class_of_interest=class_of_interest,
                                                            grid_resolution=grid_resolution,
                                                            ax=ax,
                                                            levels=class_probabilities)
        if not class_probabilities is None:
            probas_dsp.ax_.clabel(probas_dsp.surface_, 
                                  inline=True, 
                                  colors=class_probabilities_colors)

    class_ = np.unique(y_true)
    for c in class_:
        idx = y_pred == c
        decision_bound_dsp.ax_.scatter(x[idx][x.columns[0]], 
                                       x[idx][x.columns[1]], 
                                       color=[lut[c] for c in y_true[idx]], # color == true class
                                       label=classes[c], # label == predicted class
                                       marker=markers[c], # marker == predicted class
                                       edgecolor="black")

    if add_legend:
        # couleurs == true class
        true_labels = [Line2D([], [], color=lut[c], label=classes[c]) for c in class_]
        true_labels = plt.legend(handles=true_labels, title="True labels", loc="lower left", bbox_to_anchor=(1, 0.5))
        plt.gca().add_artist(true_labels)

        # marker + label == predicted class
        pred_labels = [plt.scatter([], [], color="black", marker=markers[c], label=classes[c]) for c in class_]
        pred_labels = plt.legend(handles=pred_labels, title="Predicted labels", loc="upper left", bbox_to_anchor=(1, 0.5))