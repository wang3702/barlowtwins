import seaborn as sns
import matplotlib.pylab as plt

def plot_2d_heatmap(save_path,input):
    """
    Args:
        save_path:
        input:

    Returns:

    """
    plt.cla()
    plt.clf()
    ax = sns.heatmap(input, linewidth=0.5)
    plt.savefig(save_path)