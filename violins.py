import json
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def violins(kernel):
    print(f"Plotting {kernel} violins")
    if kernel == "dedisp":
        performance = "GB/s"
    else:
        performance = "GFLOP/s"

    #devices = [
    #    ("AMD Instinct MI50", f'cache_files/{kernel}_MI50.json'),
    #    ("AMD Radeon PRO W6600", f'cache_files/{kernel}_W6600.json'),
    #    ("AMD Instinct MI250X", f'cache_files/{kernel}_MI250X.json'),
    #    ("NVIDIA RTX A4000", f'cache_files/{kernel}_A4000.json'),
    #    ("NVIDIA A100-PCIE-40GB", f'cache_files/{kernel}_A100.json')
    #]
    devices = [
        ("AMD W6600", f'cache_files/{kernel}_W6600.json'),
        ("AMD MI250X", f'cache_files/{kernel}_MI250X.json'),
        ("NVIDIA A4000", f'cache_files/{kernel}_A4000.json'),
        ("NVIDIA A100", f'cache_files/{kernel}_A100.json')
    ]

    data_df = pd.DataFrame(columns=['Device', performance])
    perf_data_df = pd.DataFrame(columns=['Device', performance])

    for device, file_path in devices:
        with open(file_path, 'r') as f:
            device_data = json.load(f)
        perf_data = []
        for v in device_data["cache"].values():
            try:
                perf_data.append(v[performance])
            except KeyError:
                pass
        max_value = max(perf_data)
        min_value = 0 #min(perf_data)
        range_value = max_value - min_value
        normalized_perf_data = [(val - min_value) / range_value for val in perf_data]
        df_norm_perf = pd.DataFrame({performance: normalized_perf_data, 'Device': device})
        df_perf = pd.DataFrame({performance: perf_data, 'Device': device})
        data_df = pd.concat([data_df,   df_norm_perf], ignore_index=True)
        perf_data_df = pd.concat([perf_data_df, df_perf], ignore_index=True)

    print(f"Statistical values of {kernel}:")
    statistics_df = perf_data_df.groupby('Device', sort=False).describe()
    statistics_df = statistics_df.round(2)
    statistics_df[performance,"normalized 50%"] = statistics_df[performance,'50%'] / statistics_df[performance,'max']
    statistics_df[performance,"normalized median"] = statistics_df[performance,'mean'] / statistics_df[performance,'max']
    print(statistics_df)

    font_size = 22
    sns.set_theme(style="whitegrid", palette="muted")
    #sns.set_context("paper", rc={"font.size":10,"axes.titlesize":7,"axes.labelsize":12})
    sns.set(font_scale = 1)

    fig, ax = plt.subplots(figsize=(9, 3))

    sns.violinplot(
            data=data_df,
            x='Device',
            y=performance,
            hue='Device',
            ax=ax,
            density_norm='width',
            #common_norm=True,
            gap=0,
            bw_adjust=.3)

    ax.set_ylabel(f"Normalized {performance}") #, fontsize=font_size)
    ax.set_xlabel("") #, fontsize=font_size)
    if kernel == "dedisp":
        ax.set_title('Dedispersion Tuning Search Space') #, fontsize=font_size)
    else:
        ax.set_title(f'{kernel.capitalize()} Tuning Search Space') #, fontsize=font_size)
    ax.set_ylim(bottom=0)
    ax.set_ylim(top=1)
    ax.tick_params(axis='x') #, labelsize=font_size)
    ax.tick_params(axis='y') #, labelsize=font_size)
    plt.savefig(f'plots/violins_{kernel}_normalized.pdf', format="pdf", bbox_inches='tight')
    plt.savefig(f'plots/violins_{kernel}_normalized.png', format='png', bbox_inches='tight')

if __name__ == "__main__":
    arg1 = sys.argv[1]
    violins(arg1)
