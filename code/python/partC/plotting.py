import os, inspect
import matplotlib.pyplot as plt

def save_plot(str):
    """
    Defines where and how to save the plots with the results from part C
    """
    def define_plots_folder_path():
        """
        Defines the absolute path of the folder where plots are saved, according to its relative path to this file
        """
        filename = inspect.getframeinfo(inspect.currentframe()).filename
        path = os.path.dirname(os.path.abspath(filename))
        splitFolderName = 'code/'
        splitPath = path.split(splitFolderName)
        plotPath = splitPath[0] + 'report/FiguresC/'
        return plotPath

    plotPath = define_plots_folder_path()
    plt.rcParams['figure.figsize'] = 3.0, 3.0
    plt.rcParams.update({'font.size': 8})
    plt.savefig(plotPath + "/" + str + ".pdf", bbox_inches='tight')


def plot_lower_bound(L_vec, plt_color, plt_name, bool_plt_saved):
    """
    Plots the results from part C
    """
    plt.figure()
    plt.plot(L_vec, color= plt_color)
    plt.xlabel('Iteration, $it$')
    plt.ylabel('Lower bound value, ${L}$')
    plt.legend(['${L}$'])
    title = 'Lower bound evaluation for location: ' + plt_name
    if bool_plt_saved:
        save_plot(plt_name)
    else:
        plt.title(title)