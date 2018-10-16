import os 
import argparse 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 


parser = argparse.ArgumentParser()
parser.add_argument('--res_dir', default="predictions", required=True, help="Root directory of the predictions from various models")
parser.add_argument('--cnames', default=None, nargs='+', help='Names of the confusion matrices to be selected for analysis')
parser.add_argument('--columns', default=['base'], nargs='+', help='List of the names of columns required for the plots')
parser.add_argument('--plot_path', required=True, help='path to save the resulting plot')



# def plot_barplot(dataframe, columns, save_dir):
#     """
#     Plot a barplot that contains a comparison of 
#     how well the model predicts classes.

#     Args:
#     DataFrame: A dataframe containing: [class, model1, model2....modeln]
#     columns: The columns of interest 
#     save_dir: Path to save the plots 
#     """
#     pass 

class BarPlot(object):
    def __init__(self, dataframe, x_label, y_label, y_lim):
        self._df = dataframe 
        self._xlab = x_label 
        self._ylab = y_label 
        self._y_lim = y_lim 
    
    def plot(self):
        """
        Plot the barplot with different metrics. 
        """
        sns.set_context("paper", font_scale=0.75) 
        sns.set_style("darkgrid")

        # sns.set_palette("husl")
        melted_df = self._df.melt('class', var_name='model_type')
        bplot = sns.catplot(x='class', y='value', hue='model_type', data=melted_df, kind="bar")
        bplot.set_axis_labels(self._xlab, self._ylab)
        # bplot.set(ylim=self._y_lim)
        
        return bplot


def create_dataframe(data_lst, columns, diag=True):
    """
    Creates a dataframe out of the given data list. 
    If diag is true then only selects the diagonal elements.

    Args:
    data_lst: A list of np.ndarrays
    columns: A list of strings indicating the columns for the dataframe 
    diag: Bool: To select diagonals or not.
    """
    df  = None
    if diag:
        diag_lst = [np.diag(x) for x in data_lst]
        classes = np.linspace(0, 9, num=10)

        diag_arr = np.vstack([classes, np.vstack(diag_lst)])
        print(diag_arr.shape)
        df = pd.DataFrame(data=diag_arr.T, columns=columns)
    return df 






if __name__ == '__main__':
    args = parser.parse_args()
    if not os.path.exists(args.res_dir):
        raise NotADirectoryError("The given directory doesn't exist")
    
    if len(args.cnames) == 0:
        raise IndexError("Need atleast one confusion matrix to plot data")
    

    data_lst = []
    for cname in args.cnames:
        print(cname)
        data_lst.append(np.loadtxt(os.path.join(args.res_dir, cname+'.txt')))
    
    data_frame = create_dataframe(data_lst, columns=args.columns)
    x_lab = 'Classes'
    y_lab = 'Correct Predictions'
    y_lim = (0, 10000)
    bplot = BarPlot(data_frame, x_lab, y_lab, y_lim)
    plot = bplot.plot()
    plt.savefig(args.plot_path)
    
    
