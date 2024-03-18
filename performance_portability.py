import json
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
import scipy
import numpy as np

cmap = "inferno"

def subplot_portability_data(rows, cbar=False):
    xlabels = [
        "W6600",
        "MI250X",
        "A4000",
        "A100",
    ]

    ylabels = xlabels + [
        "AMD",
        "NVIDIA",
        "All"
    ]

    sns.heatmap(rows, cmap=cmap, annot=True, yticklabels=ylabels, xticklabels=xlabels, vmin=0, cbar=cbar)
    plt.xticks(rotation=0)

def subplot_portability(kernel):
    print(f"Plotting {kernel} portability")
    if kernel == "dedisp":
        performance = "GB/s"
    else:
        performance = "GFLOP/s"

    devices = [
        #("AMD Instinct MI50", f'cache_files/{kernel}_MI50.json'),
        ("AMD Radeon PRO W6600", f'cache_files/{kernel}_W6600.json'),
        ("AMD Instinct MI250X", f'cache_files/{kernel}_MI250X.json'),
        ("NVIDIA RTX A4000", f'cache_files/{kernel}_A4000.json'),
        ("NVIDIA A100-PCIE-40GB", f'cache_files/{kernel}_A100.json')
    ]

    data_per_device = dict()
    configurations = set()

    for device, file_path in devices:
        with open(file_path, 'r') as f:
            device_data = json.load(f)

        tune_params_keys = device_data["tune_params_keys"]
        perf_data = dict()

        for v in device_data["cache"].values():
            try:
                config = tuple(v[param_key] for param_key in tune_params_keys)
                value = v[performance]
                perf_data[config] = value
            except KeyError:
                pass

        configurations |= set(perf_data.keys())

        best_performance = max(perf_data.values())
        data_per_device[device] = dict((k, v / best_performance) for k, v in perf_data.items())

    print(f"{len(configurations)} configurations found")

    df = pd.DataFrame({"configuration": list(configurations)})
    for device in data_per_device:
        df[device] = [data_per_device[device].get(config) for config in df["configuration"]]

    before = len(df)
    df = df.dropna(ignore_index=True)
    after = len(df)

    print(f"{before - after} records removed")

    all_devices = [device for device in data_per_device]
    amd_devices = [device for device in all_devices if "AMD" in device]
    nvidia_devices = [device for device in all_devices if "AMD" not in device]

    df["score"] = df[all_devices].apply(scipy.stats.hmean, axis=1)
    df["score_amd"] = df[amd_devices].apply(scipy.stats.hmean, axis=1)
    df["score_nvidia"] = df[nvidia_devices].apply(scipy.stats.hmean, axis=1)

    configs = [
        np.argmax(df["AMD Radeon PRO W6600"]),
        np.argmax(df["AMD Instinct MI250X"]),
        np.argmax(df["NVIDIA RTX A4000"]),
        np.argmax(df["NVIDIA A100-PCIE-40GB"]),
        np.argmax(df["score_amd"]),
        np.argmax(df["score_nvidia"]),
        np.argmax(df["score"]),
    ]

    print(df.iloc[configs])
    print(df.iloc[configs][all_devices])

    rows = np.array([
        [df[device][config] for device in all_devices]
        for config in configs
    ])


    subplot_portability_data(rows)
    return rows




def plot_portability():
    scale = 0.75
    plt.subplots(1, 4, figsize=(15 * scale, 4 * scale), sharey=True)

    plt.subplot(141)
    plt.title("Convolution")
    data = subplot_portability("convolution")
    plt.xlabel("... applied to...")
    plt.ylabel("Configuration with maximum\nperformance portability score for...")

    plt.subplot(142)
    plt.title("Hotspot")
    data += subplot_portability("hotspot")
    plt.xlabel("... applied to...")

    plt.subplot(143)
    plt.title("Dedispersion")
    data += subplot_portability("dedisp")
    plt.xlabel("... applied to...")

    plt.subplot(144)
    plt.title("GEMM")
    data += subplot_portability("gemm")
    plt.xlabel("... applied to...")

    #plt.subplot(155)
    #norm =  matplotlib.colors.Normalize(0, 1)
    #plt.gcf().colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=plt.gca())

    plt.tight_layout()
    plt.savefig(f"plots/portability.eps")

    avg = data / 4
    plt.figure(figsize=(6 * scale, 4 * scale))
    plt.title("Average")
    subplot_portability_data(avg, cbar=True)
    plt.xlabel("... applied to...")
    plt.ylabel("Configuration with maximum\nperformance portability score for...")

    plt.tight_layout()
    plt.savefig(f"plots/portability_avg.eps")


if __name__ == "__main__":
    plot_portability()
