import numpy as np
import csv
import json
import os
import itertools
import warnings
import matplotlib.pyplot as plt
import networkx as nx
import pickle

import bloopy
from bloopy.individual import individual, continuous_individual
import bloopy.utils as utils
import bloopy.analysis.analysis_utils as anutil
import bloopy.analysis.critical_points as critpts
import bloopy.analysis.FFG
import gpu_utils

def compute_and_analyze():
    np.set_printoptions(precision=4)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"

    # Read file
    # We do hyperparameter tuning on GTX 1080Ti files
    data_path = root_dir + 'AutoTuning_AMD_vs_Nvidia_GPUs/processed_cache_files/'
    #data_path = '/var/scratch/mli940/'

    convolution_files = [
        #'convolution_MI50_processed.json',
        'convolution_MI250X_processed.json',
        'convolution_W6600_processed.json',
        'convolution_A4000_processed.json',
        'convolution_A100_processed.json'
    ]

    hotspot_files = [
        #'hotspot_MI50_processed.json',
        'hotspot_MI250X_processed.json',
        'hotspot_W6600_processed.json',
        'hotspot_A4000_processed.json',
        'hotspot_A100_processed.json'
    ]

    dedisp_files = [
        #'dedisp_MI50_processed.json',
        'dedisp_MI250X_processed.json',
        'dedisp_W6600_processed.json',
        'dedisp_A4000_processed.json',
        'dedisp_A100_processed.json'
    ]

    gemm_files = [
        #'gemm_MI50_processed.json',
        'gemm_MI250X_processed.json',
        'gemm_W6600_processed.json',
        'gemm_A4000_processed.json',
        'gemm_A100_processed.json'
    ]

    for files in (gemm_files, convolution_files, hotspot_files, dedisp_files):
        for filename in files:
            print(f"Computing and analyzing {filename} FFG")
            with open(data_path + filename, 'r') as myfile:
                data=myfile.read()
            data = json.loads(data)

            print("\nDevice: " + str(data['device_name']))
            print("Kernel name: " + str(data['kernel_name']))
            print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')

            # Pre-process the search space
            searchspace_orig = data['tune_params']
            searchspace = utils.clean_up_searchspace(searchspace_orig)
            #print("Processed search space:", searchspace)

            ### Calculate bitstring size
            bsize = utils.calculate_bitstring_length(searchspace)
            #print("Size of bitstring after pre-processing:", bsize)

            ### Number of variables
            nr_vars = len(searchspace.keys())

            # Construct the GPU tuning space
            GPU_space = gpu_utils.GPU_tuning_space(searchspace, searchspace_orig, data['cache'])

            disc_space = utils.discrete_space(GPU_space.get_runtime, searchspace)


            ### Compute optimal fitness for reference
            if "dedisp" in data["kernel_name"]:
                performance = "GB/s"
            else:
                performance = "GFLOP/s"

            best_fit = 100000000
            bestkey = None
            for k in list(data['cache'].keys()):
                try:
                    # HACK: bloopy assumse that we want to minimize "time", however, we
                    # instead want to maximize "performance". To solve this, we
                    # set the "time" column to "1/performance" as a 'workaround'.
                    fake_time = 1 / float(data['cache'][k][performance])
                    data['cache'][k]["time"] = fake_time

                    if fake_time < best_fit:
                        best_fit = fake_time
                        bestkey = k
                except e:
                    print(e)
                    pass
            print("Optimal settings in cache are:", bestkey, "with time {0:.4f}".format(best_fit))
            print("There are", len(data['cache'].keys()), "keys in the searchspace")

            ###  <<<  ANALYZING SEARCH SPACES  >>>
            #method = 'circular'
            method = 'bounded'
            #method = 'Hamming'
            boundary_list = utils.generate_boundary_list(searchspace)
            indiv = individual(bsize, boundary_list=boundary_list)

            ## Find the global minimum
            best_key_bs = gpu_utils.convert_gpusetting_to_bitidxs(bestkey, boundary_list, searchspace_orig)
            utils.set_bitstring(indiv, list(best_key_bs))
            glob_fit = disc_space.fitness(indiv.bitstring)
            print("Global minimum:", bestkey, "with fitness", glob_fit)


            ## Loop through the space and assign point types to each point.
            ## Also build the space dictionary.
            nidxs_dict = anutil.build_nodeidxs_dict(boundary_list, disc_space.fitness, bsize)
            tot, minimas, maximas, saddles, regulars, spacedict = critpts.classify_points(bsize, boundary_list, nidxs_dict, method=method)

            idxs_to_pts = anutil.indices_to_points(spacedict)
            print(tot, minimas, maximas, saddles, regulars)

            ###   COMPUTE FFG   ###
            G = bloopy.analysis.FFG.build_FFG(nidxs_dict, boundary_list, method=method)


            ###   ANALYZE FFG   ###
            graph_name = "FFG_data/FFG_" + method + "_" + filename[:-5] + ".txt"
            ## Check graph properties such as cycles
            globopt_idx = spacedict[tuple(best_key_bs)][2]
            print("Global optimum is node:", globopt_idx)
            print(len(G.nodes()), "nodes,", len(G.edges()), "edges, in search space graph")

            ## Calculate centralities
            #centrality = "degree"
            #centrality = "katz"
            centrality = "pagerank"
            #centrality = "closeness"
            if centrality == "degree":
                # Degree centrality can be seen as the immediate risk of a node for catching whatever is flowing through the network.
                # So a high degree centrality for global optimum means there is
                # a better chance of ending up there.
                centrality_dict = nx.algorithms.centrality.degree_centrality(G)
                print("Degree centrality of global optimum:", centrality_dict[globopt_idx])
            elif centrality == "eigen":
                # Eigen vector centraliy is similar, it is a measure of the influence
                # of a node in a network.
                centrality_dict = nx.eigenvector_centrality_numpy(G.reverse())
                print("Eigen vector centrality of global optimum:", centrality_dict[globopt_idx])
            elif centrality == "katz":
                # Katz centrality is a variant of eigenvector, look up details
                centrality_dict = nx.katz_centrality_numpy(G)
                print("Katz vector centrality of global optimum:", centrality_dict[globopt_idx])
            elif centrality == "secondorder":
                # The second order centrality of a given node is the standard
                # deviation of the return times to that node of a perpetual
                # random walk on G.
                centrality_dict = nx.algorithms.centrality.second_order_centrality(G)
                print("Second order centrality of global optimum:", centrality_dict[globopt_idx])
            elif centrality == "closeness":
                centrality_dict = nx.algorithms.centrality.closeness_centrality(G)
                print("Closeness centrality of global optimum:", centrality_dict[globopt_idx])
            elif centrality == "pagerank":
                centrality_dict = nx.algorithms.link_analysis.pagerank_alg.pagerank(G)
                print("Pagerank centrality of global optimum:", centrality_dict[globopt_idx])
            else:
                raise Exception("Unknown centrality type")
            centr_name = "FFG_data/propFFG_centrality_" + centrality + "_" + method + "_" + filename[:-5] + ".csv"
            percs = np.arange(0.0, 0.16, 0.01).tolist()
            centralities = [["Percentage","proportion_centr","sum_accept_centr", "tot_centr", "minima_centr", "nr_of_nodes"]]
            for perc in percs:
                acceptable_minima = critpts.strong_local_minima(perc, glob_fit, spacedict)
                accept_centr, tot_centr, minima_centr = bloopy.analysis.FFG.average_centrality_nodes(centrality_dict, acceptable_minima, spacedict, idxs_to_pts)
                prop_centr = accept_centr/float(minima_centr)
                centralities.append([perc, prop_centr, accept_centr, tot_centr, minima_centr, len(centrality_dict.values())])
                print("Proportion of centrality of strong local minima", perc, ":",prop_centr)
            # Save to CSV file
            with open(centr_name, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(centralities)

            #NOTE: (Un)comment continue if you only want to compute pagerank centralities
            continue

            ## Plot the graph with NetworkX
            color_map = []
            size_map = []
            threshold = 0.75
            for node in G:
                pt = idxs_to_pts[node][0]
                if spacedict[pt][0] == 1:
                    fit = spacedict[pt][1]
                    color_map.append(glob_fit/fit)
                    siz = 5
                    if glob_fit/fit > threshold:
                        siz += 4*8*(glob_fit/fit - threshold)
                    size_map.append(siz)
                else:
                    fit = spacedict[pt][1]
                    color_map.append(glob_fit/fit)
                    size_map.append(0.7)

            # Define colormap with transparency
            from matplotlib.colors import ListedColormap
            #cmap = plt.get_cmap('viridis')
            cmap = plt.get_cmap('winter')

            # Get the colormap colors
            my_cmap = cmap(np.arange(cmap.N))

            # Set alpha
            my_cmap[:,-1] = np.linspace(0.5, 1.0, cmap.N)

            # Create new colormap
            my_cmap = ListedColormap(my_cmap)

            # Some technicalities for plotting
            cf = plt.gcf()
            cf.set_facecolor("w")
            if cf._axstack is None:
                ax = cf.add_axes((0, 0, 1, 1))
            else:
                ax = cf.gca()

            if nx.is_directed(G):
                arz = 2
                arst = '-|>'
                wdth = 0.001
                alp = 0.1
                #pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
                #nx.draw(G, pos=pos, node_color=color_map, arrowsize=arz, node_size=size_map, with_labels=False, arrows=True,arrowstyle=arst, font_size=1, cmap=my_cmap, vmin=0.75, vmax=1.0, width=wdth)

                #nx.draw_kamada_kawai(G, arrows=True, arrowstyle=arst, arrowsize=arz, node_size=size_map, node_color=color_map, font_size=1, with_labels=False, cmap=my_cmap, vmin=0.75, vmax=1.0, width=wdth)
                pos = nx.kamada_kawai_layout(G)
                pltnodes = nx.drawing.nx_pylab.draw_networkx_nodes(G, pos, ax=ax, node_size=size_map, node_color=color_map, cmap=my_cmap, vmin=0.75, vmax=1.0, linewidths=0.0)
                nx.drawing.nx_pylab.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle=arst, arrowsize=arz, width=wdth, node_size=size_map, alpha=alp)

                plt.draw_if_interactive()
                ax.set_axis_off()
                plt.draw_if_interactive()

                # Draw colorbar
                cbar = plt.colorbar(pltnodes, shrink=0.5)
                cbar.set_label('Fraction of optimal fitness', rotation=270, labelpad=15)

            plt.axis('off')
            plt.draw()
            plt.savefig(graph_name[:-4] + ".pdf")
            plt.clf()
            del G
            del spacedict
            del centrality_dict
            del acceptable_minima
            del idxs_to_pts


if __name__ == '__main__':
    compute_and_analyze()
