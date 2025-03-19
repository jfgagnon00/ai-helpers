import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from ipywidgets import Output, Layout, HBox

def confusion_matrix_analysis(title_suffix, 
                              y_true, 
                              y_pred, 
                              target_names, 
                              normalize="pred",
                              figsize=(4, 4),
                              **plot_kwargs):
    """
    Affiche matrice de confusion et rapport detaille de classification
    ATTENTION: 
        normalize == "true" veut dire que c'est recall qu'on a sur la diagonale
        normalize == "pred" veut dire que c'est precision qu'on a sur la diagonale
        normalize == None veut dire que normalization ne sera pas affiche
    """
    matrix_ = Output()
    with matrix_:
        fig = plt.figure(figsize=figsize)

        subplot_idx = 121 if not normalize is None else 111
        
        plt.subplot(subplot_idx)
        conf_matrix = confusion_matrix(y_true, y_pred, normalize=None)
        conf_matrix_plt = ConfusionMatrixDisplay(conf_matrix,
                                                    display_labels=target_names)
        conf_matrix_plt.plot(ax=plt.gca(), 
                                values_format="d",
                                **plot_kwargs)
        conf_matrix_plt.im_.colorbar.remove()

        if not normalize is None:
            plt.title("Non normalisee")

            conf_matrix_norm = confusion_matrix(y_true, y_pred, normalize=normalize)

            plt.subplot(122)
            conf_matrix_plt = ConfusionMatrixDisplay(conf_matrix_norm, 
                                                        display_labels=target_names)
            conf_matrix_plt.plot(ax=plt.gca(), 
                                values_format=".2f",
                                **plot_kwargs)
            conf_matrix_plt.im_.colorbar.remove()
            plt.title(f"Normalisee - {normalize}")

        plt.suptitle(f"Matrice de confusion - {title_suffix}")
        plt.tight_layout()
        plt.close()
        display(fig)

    report_ = Output()
    with report_:
        report = classification_report(y_true, y_pred, target_names=target_names)
        print("Rapport par classe")
        print(report)

    display( HBox([matrix_, report_], 
                  layout=Layout(align_items="center")) )
