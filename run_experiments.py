import compute_and_analyze_FFGs
import process_cache_files
import plot_centralities
import violins

if __name__ == '__main__':
    #NOTE: uncomment if you have new cache files in cache_files dir
    #process_cache_files.process()
    #compute_and_analyze_FFGs.compute_and_analyze()
    plot_centralities.plot()
    for kernel in ("convolution", "hotspot", "dedisp"):
        violins.violins(kernel)