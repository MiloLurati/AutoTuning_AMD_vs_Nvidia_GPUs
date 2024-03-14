import json
import sys
import os

kernels = ["dedisp", "hotspot", "convolution", "gemm"]
devices = ["W6600", "MI250X", "A4000", "A100"]
path = path = os.path.dirname(os.path.realpath(__file__)) + "/cache_files/"


def print_top_configs(kernel_name):

    for dev in devices:
        filename = path + kernel_name + "_" + dev + ".json"

        print(r"\begin{minipage}{0.249\columnwidth}\scriptsize\centering")
        print(dev + r" \\")
        print(r"""\begin{tabular}{lllllll}
\midrule""")

        with open(filename) as f:
            data = json.load(f)

        tune_param_keys = data["tune_params_keys"]
        tune_params = data["tune_params"]
        objective = data["objective"]

        #print(filename)
        #print("keys:")
        for key in list(tune_param_keys):
            if len(tune_params[key]) > 1:
                pass
                #print(f" - {key}: {tune_params[key]}")
            else:
                tune_param_keys.remove(key)

        #filter out invalid configs
        records = [record for record in list(data["cache"].values()) if isinstance(record[objective], float)]

        # prefer GFLOP/s or GB/s as objective over time
        if "GFLOP/s" in records[0]:
            objective = "GFLOP/s"
        if "GB/s" in records[0]:
            objective = "GB/s"

        #sort on objective
        #print(f"sort on {objective}")
        records.sort(key=lambda p : p[objective], reverse=True)

        for i in range(5):
            #print(records[i][objective], best, best/records[i][objective])
            records[i]["best"] = "{:.2f}".format(records[i][objective])

        for record in records[:5]:
            print(" & ".join(str(record[key]) for key in tune_param_keys+["best"]) + f" \\\\ % \t time: {record['time']} \t {objective}: {record[objective]}")

        print(r"""\midrule
\end{tabular} \\
\end{minipage}%""")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in kernels:
        print_top_configs(sys.argv[1])
    else:
        print("Usage: python top_configurations.py [kernel_name]")
        print("  kernel_name options:", ", ".join(kernels))
