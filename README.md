# Data and Plotting scripts for "Comparing the Impact & Difficulty of Auto-Tuning on AMD and Nvidia GPUs"

This repository contains the cached GPU tuning data, the Python plotting scripts, and the kernels and kernel scripts for the paper "Comparing the Impact & Difficulty of Auto-Tuning on AMD and Nvidia GPUs" by Milo Lurati and Ben van Werkhoven.

## Installation

The code makes use of the [BlooPy](https://github.com/schoonhovenrichard/BlooPy) Python package. Please ensure the latest version is installed.

```
pip install bloopy
```

To re-create the plots requires the seaborn package.

```
pip install seaborn
```

To re-create the cache files you will need [Kernel Tuner](https://kerneltuner.github.io/kernel_tuner/stable/) with [HIP](https://docs.amd.com/projects/HIP/en/docs-5.3.0/how_to_guides/install.html).

## Running the experiments

For anyone interested in repeating the experiments described in the paper the following describes how to run the experiments.

The main file for running experiments is ```run_experiments.py```. It can be used as follows:

```
python run_experiments.py
```

In ```run_experiments.py``` by default, there are two lines of code commented out. These lines process the cache files and compute and analyze the FFGs. As these results are included by default in the repository, ```run_experiments.py``` will only plot the violin plots and centrality plots and calculate the statistical values of the search spaces.
