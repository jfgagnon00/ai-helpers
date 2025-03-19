import pandas as pd

from IPython.display import display, HTML


def display_html(html_message):
    """
    Utilitaire pour afficher message contenant encoding HTML
    """
    display(HTML(html_message))

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

def align_left_df(styler, subset=None, center_header=True):
    """
    Align all subset columns of dataframe to left. Optionaly center
    the column headers
    """
    if isinstance(styler, pd.DataFrame):
        styler = styler.style

    styler_props = {'text-align': 'left'}
    styler.set_properties(subset=subset, **styler_props)

    if center_header:
        styles = {s:[{"selector": "th",
                      "props": [('text-align', 'center')]}] for s in subset}
        styler.set_table_styles(styles, overwrite=False)

    return styler

def caption_df(styler,
               caption,
               caption_color="black",
               caption_size="120%"):
    """
    Add a caption to a dataframe
    """
    caption_styles = [dict(selector="caption",
                            props=[("color", caption_color),
                                    ("font-size", caption_size),
                                    ("font-weight", "bold")])]

    if isinstance(styler, pd.DataFrame):
        styler = styler.style

    styler.set_caption(caption).set_table_styles(caption_styles, overwrite=False)

    return styler

def horizontify(*dataframes, padding=25):
    """
    Ecapsule chaque element de dataframes avec <div></div>
    """
    html = ""

    for df in dataframes:
        html += f"<div style='float:left;padding: 0px {padding}px 0px 0px;'>{df.to_html()}</div>"

    return html