import os, inspect
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 2.5, 2.5
plt.rcParams.update({'font.size': 7})

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
        save_plot('Lbound_' + plt_name)
    else:
        plt.title(title)

def plot_responsibilities(Nk_vec, plt_color, plt_name, bool_plt_saved):
    plt.figure()
    plt.plot(Nk_vec, color=plt_color, marker='o', linestyle='')
    plt.xlabel('Mixand k')
    plt.ylabel('Responsibility value $N_k$')
    title = 'Responsibilities of GM model for location: ' + plt_name
    if bool_plt_saved:
        save_plot('Nk_norm_' + plt_name)
    else:
        plt.title(title)
    return