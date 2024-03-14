import json
import sys

def print_top_configs(filename):
    with open(filename) as f:
        data = json.load(f)

    tune_param_keys = data["tune_params_keys"]
    tune_params = data["tune_params"]

    print(filename)
    print("keys:")
    for key in list(tune_param_keys):
        if len(tune_params[key]) > 1:
            print(f" - {key}: {tune_params[key]}")
        else:
            tune_param_keys.remove(key)

    if "dedisp" in data["kernel_name"]:
        performance = "GB/s"
    else:
        performance = "GFLOP/s"

    records = list(data["cache"].values())
    records.sort(key=lambda p: -p[performance] if isinstance(p.get(performance), float) else float("inf"))

    for record in records[:5]:
        line = " & ".join(str(record[key]) for key in tune_param_keys)
        #line += f" \\\\ % time: {record['time']}"
        line += f" & {record[performance]:.2f} \\\\ %"
        while len(line) < 40: line += ' '
        line += f"time: {record['time']}"
        print(line)
    print()



if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print_top_configs(filename)
