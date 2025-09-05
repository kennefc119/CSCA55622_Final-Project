import scipy as sp
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import statsmodels.formula.api as smf
import statsmodels.api as sm

#########Data Analysis & Visualisation###############################################################
#Pairplot Visual investigation
def pair_plotting(df,img_file_name = "pairplot.png", trim = 10,samples = 1000,label_font_size = 20,dot_size = 40):
    """Create a pairplot for the given DataFrame.
    The plot shows the graphical relationship between Y and all X.
    If there is a linearity (diagonal line), it indicates a potential linear relationship.
    It can help find Y and its significant X, while also discover hidden correlation between X

    Args:
        df (pd.DataFrame): The input DataFrame.
        img_file_name (str, optional): The filename for saving the plot. Defaults to "pairplot.png".
        trim (int, optional): The number of columns to include in the pairplot. Defaults to 10.
        samples (int, optional): The number of samples to include in the pairplot. Defaults to 1000.
        label_font_size (int, optional): The font size for the labels. Defaults to 20.
    """
    partial_df = df.iloc[:, :trim]
    graph = sns.pairplot(partial_df.sample(samples), 
                diag_kind='kde',
                corner=False,
                kind='scatter',
                plot_kws={'s': dot_size},
                )

    for ax in graph.axes.flatten():
        if ax:
            ax.set_xticks([])
            ax.set_yticks([])
    for i, ax in enumerate(graph.axes[-1, :]):
        if ax:
            ax.set_xlabel(ax.get_xlabel(), fontsize=label_font_size)

    for i, ax in enumerate(graph.axes[:, 0]):
        if ax:
            ax.set_ylabel(ax.get_ylabel(), fontsize=label_font_size)

    graph.savefig(img_file_name, facecolor='white', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

#Visualisation of parameters correlation - Heat map
def heatmap_correlation(df,img_file_name = "heatmap.png", label_font_size = 8.5,annotation_size = 6.5):
    """Create a heatmap to visualize the correlation matrix.
    The closer the value to 1, means there is a higher correlation.

    Args:
        df (pd.DataFrame): The input DataFrame.
        img_file_name (str, optional): The filename for saving the plot. Defaults to "heatmap.png".
        label_font_size (float, optional): The font size for the labels. Defaults to 8.5.
        annotation_size (float, optional): The font size for the annotations. Defaults to 6.5.
    """
    graph = sns.heatmap(df.corr(),cmap='YlGnBu',
                center=0, 
                annot=True, 
                fmt=".1f",
                annot_kws={"size":annotation_size},
            linewidths=0.5,
            linecolor='white',
            square=True,
            cbar_kws={"shrink": 0.8},
            )

    plt.xticks(fontsize=label_font_size)
    plt.yticks(fontsize=label_font_size)
    cbar = plt.gca().collections[0].colorbar
    cbar.ax.tick_params(labelsize=label_font_size)
    graph.figure.savefig(img_file_name, facecolor='white', bbox_inches='tight')
    plt.tight_layout()
    plt.show()

#########Linear Modelling###############################################################
#ranking individual parameters 
def ranking_linear_models_individual(df,target_Y_para):
    """Rank individual linear regression models based on adjusted R-squared.

    Args:
        df (pd.DataFrame): The input DataFrame containing features and target variable.

    Returns:
        dict: A dictionary with feature names as keys and their adjusted R-squared values as values.
        list: A list of the top three features based on adjusted R-squared.
    """
    result_dict = {}
    for paraX in df.columns:
        if paraX != target_Y_para:
            model = smf.ols(f'{target_Y_para} ~ {paraX}', data=df).fit()
            adj_R2 = model.rsquared_adj
            result_dict[paraX] = adj_R2
    result_dict = dict(sorted(result_dict.items(), key=lambda item: item[1], reverse=True))
    top_three_Xpara  = list(result_dict.keys())[:3]
    print (f'Top three features for predicting {target_Y_para}: {top_three_Xpara}')
    return result_dict,top_three_Xpara

#########Multilinear Modelling###############################################################
