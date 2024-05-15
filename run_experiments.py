import compute_and_analyze_FFGs
import process_cache_files
import plot_centralities
import violins
from top_configurations import print_top_configs
from performance_portability import plot_portability

if __name__ == '__main__':
    #NOTE: uncomment if you have new cache files in cache_files dir
    #process_cache_files.process()
    #compute_and_analyze_FFGs.compute_and_analyze()
    print("\n==================Plotting proportion of centralities...\n")
    plot_centralities.plot()
    print("\n==================Plotting violin plots...\n")
    for kernel in ("convolution", "hotspot", "dedisp", "gemm"):
        violins.violins(kernel)
        print(f"\n==================Printing {kernel} top configurations...\n")
        print_top_configs(kernel)
    print("\n==================Plotting performance portability...\n")
    plot_portability()