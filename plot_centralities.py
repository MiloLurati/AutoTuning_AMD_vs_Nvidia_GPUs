import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import seaborn as sns
import pandas as pd

def plot_centrality(plotdf, kernel, order):
    ### Make the plots
    # Plot settings
    fm = ''
    cps = 2.0
    linesty = '-'
    import matplotlib
    #font = {'family' : 'sans-serif',
    #        'weight' : 'bold',
    #        'size'   : 34}
    #matplotlib.rc('font', **font)

    # Create the SEABORN plot
    sns.set_theme(style="whitegrid", palette="muted")
    #sns.set_context("paper", rc={"font.size":10,"axes.titlesize":7,"axes.labelsize":12})
    sns.set(font_scale = 1)

    ## DEFINE COLOUR PALETTE:
    palette ={
            #"MI50": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            "MI250X": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            "W6600": (0.2980392156862745, 0.4470588235294118, 0.6901960784313725),
            "A4000": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804),
            "A100": (0.7686274509803922, 0.3058823529411765, 0.3215686274509804)
            }
    markers = {
            #"MI50": "<",
            "MI250X": "^",
            "W6600": "v",
            "A4000": "o",
            "A100": "s",
            "AMD": "^",
            "Nvidia": "o",
            }

    linestyles = {
            #'MI50': (3, 1.25, 3, 1.25, 1.25, 1.25),
            'MI250X': (3, 1.25, 3, 1.25, 1.25, 1.25),
            'W6600': (4, 1.5),
            'A4000': "",
            'A100': (1, 1)
            }

    if kernel == "avg":
        colors = sns.color_palette()
        palette = [colors[3], colors[2]]
        group = 'vendor'
    else:
        palette = sns.color_palette()
        group = 'GPU'

    scale = 0.75
    fig, ax = plt.subplots(figsize=(9 * scale, 4 * scale))

    g = sns.lineplot(
            data=plotdf,
            y='Prop_centrality',
            x='Percentage',
            hue=group,
            hue_order=order,
            style=group,
            style_order=order,
            linewidth=2.5,
            ax=ax,
            dashes=False,
            markersize=10,
            palette=palette,
            markers=markers)
            #dashes=linestyles)

    if kernel != "avg":
        g.set_title(f"Proportion of centrality for {kernel} per GPU") #, fontdict={'fontsize': 26})
    else:
        g.set_title("Average Proportion of Centrality over all Kernels")

    g.set_xlabel("Percentage acceptable minima") #, fontsize=22)
    g.set_ylabel("Proportion of centrality") #, fontsize=22)

    legend_properties = {'size':10}
    legendMain=g.legend(prop=legend_properties)

    ax.set(yscale="log")
    ax.set_ylim((None, 1.1))

    # Add more ticks and labels on the y-axis
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(10**-.5, 1)))
    #ax.yaxis.set_major_locator(ticker.AutoLocator())
    formatter = ticker.ScalarFormatter()
    formatter.set_powerlimits((-3, 4))  # Adjust the power limits for scientific notation if needed
    formatter.set_scientific(True)  # Enable scientific notation
    formatter.set_useOffset(False)  # Disable offset notation
    #ax.yaxis.set_major_formatter(formatter)

    # Adjust the font size of the y-axis labels
    #ax.tick_params(axis='y', labelsize=10)  # Specify the desired font size

    plt.savefig(f"plots/prop_centrality_{kernel}.pdf", format="pdf", bbox_inches='tight')
    plt.savefig(f"plots/prop_centrality_{kernel}.png", format="png", bbox_inches='tight')


def plot():
    ### Get the files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"
    pth = 'AutoTuning_AMD_vs_Nvidia_GPUs/FFG_data/'
    experiment_dir = root_dir + pth #+ 'bounded_pagerank_centrality/'
    GPUs = ["W6600", "MI250X", "A4000", "A100"]

    results = []

    for kernel in ("convolution", "hotspot", "dedisp", "gemm"):
        print(f"Plotting {kernel} centralities")
        file_dir = experiment_dir
        exper_files = [f for f in os.listdir(file_dir) if os.path.isfile(os.path.join(file_dir, f))]
        columns = ["GPU", "Percentage", "Prop_centrality", "Sum_accept_centr", "Tot_centr", "Minima_centr", "Tot_nodes"]
        dataframe_lst = []
        for f in exper_files:
            raw = f.split("_")
            if "centrality" not in raw:
                continue
            if kernel not in raw:
                continue
            if "MI50" in raw:
                continue

            # Find GPU
            gpu = None
            if "MI250X" in raw:
                gpu = "MI250X"
            else:
                gpu = "_".join(raw[5:-1])
            if gpu is None:
                print(f.split("_"))
                raise Exception("Something wrong")

            # Open the file
            with open(file_dir + f, 'r') as read_obj:
                csv_reader = csv.reader(read_obj)
                list_data = list(csv_reader)[1:]
            for dat in list_data:
                perc = float(dat[0])
                propcentr = float(dat[1])
                sumacceptcentr = float(dat[2])
                totcentr = float(dat[3])
                minimacentr = float(dat[4])
                totnodes = int(dat[5])
                entry = [gpu, perc, propcentr, sumacceptcentr, totcentr, minimacentr, totnodes]
                dataframe_lst.append(entry)

        plotdf = pd.DataFrame(dataframe_lst, columns=columns)
        plot_centrality(plotdf, kernel, GPUs)

        results.append(plotdf)

    result = pd.concat(results)
    result["vendor"] = result["GPU"].apply(lambda name: "Nvidia" if name.startswith("A") else "AMD")
    plot_centrality(result, "avg", ["AMD", "Nvidia"])

if __name__ == '__main__':
    plot()
