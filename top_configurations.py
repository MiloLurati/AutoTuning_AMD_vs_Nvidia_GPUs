import json
import sys

def print_top_configs(filename):
    with open(filename) as f:
        data = json.load(f)

    tune_param_keys = data["tune_params_keys"]
    tune_params = data["tune_params"]

    print("keys:")
    for key in list(tune_param_keys):
        if len(tune_params[key]) > 1:
            print(f" - {key}: {tune_params[key]}")
        else:
            tune_param_keys.remove(key)

    records = list(data["cache"].values())
    records.sort(key=lambda p: p["time"])

    for record in records[:5]:
        print(" & ".join(str(record[key]) for key in tune_param_keys) + f" \\\\ % time: {record['time']}")



if __name__ == "__main__":
    for filename in sys.argv[1:]:
        print_top_configs(filename)
