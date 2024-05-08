import json
import os
import copy
import statistics
import numpy as np

def process():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = "/".join(current_dir.split('/')[:-1]) + "/"

    data_path = root_dir + 'AutoTuning_AMD_vs_Nvidia_GPUs/cache_files/'
    processed_data_path = root_dir + 'AutoTuning_AMD_vs_Nvidia_GPUs/processed_cache_files/'

    FJ_files = ['convolution_A100_FJ.json']

    convolution_files = [
        'convolution_MI250X.json',
        #'convolution_MI50.json',
        'convolution_W6600.json',
        'convolution_A4000.json',
        'convolution_A100.json'
    ]

    hotspot_files = [
        'hotspot_MI250X.json',
        #'hotspot_MI50.json',
        'hotspot_W6600.json',
        'hotspot_A4000.json',
        'hotspot_A100.json'
    ]

    dedisp_files = [
        'dedisp_MI250X.json',
        #'dedisp_MI50.json',
        'dedisp_W6600.json',
        'dedisp_A4000.json',
        'dedisp_A100.json'
    ]

    gemm_files = [
        'gemm_MI250X.json',
        'gemm_W6600.json',
        'gemm_A4000.json',
        'gemm_A100.json'
    ]

    all_files = (gemm_files, convolution_files, hotspot_files, dedisp_files)

    for files in all_files:
        for filename in files:
            print(f"Processing {filename}")
            with open(data_path + filename, 'r') as myfile:
                data=myfile.read()
            data = json.loads(data)
            print(data['tune_params'])

            average_stdev = 0.0
            N = 0
            compiled_points = 0
            keys_to_pop = []
            for key, val in data['cache'].items():
                runtimeFailedConfig = False
                try:
                    meantime = float(val['time'])
                except:
                    runtimeFailedConfig = True
                if runtimeFailedConfig:
                    keys_to_pop.append(key)
                    continue
                if meantime < 1e10:
                    compiled_points += 1
                if 'times' in val.keys():
                    normalized_times = (np.array(val['times'])/float(meantime))
                    stdev = statistics.stdev(normalized_times)
                    average_stdev += stdev
                    N += 1
            for key in keys_to_pop:
                data["cache"].pop(key)
            print('Average normalized stdev of runtime:', average_stdev/float(N))
            print("Number of valid points in space:", compiled_points)

            print("Device: " + str(data['device_name']))
            print("Kernel name: " + str(data['kernel_name']))
            print("Tunable parameters: " + str(data['tune_params_keys']), end='\n\n')

            # Pre-process the search space
            searchspace = data['tune_params']
            print("There are", len(data['cache'].keys()), "keys in the searchspace")

            for k in data['cache'].keys():
                try:#Power is recorderd if it is a valid kernel setting
                    data['cache'][k].pop('power')
                except:
                    continue
                try:
                    data['cache'][k].pop('energy')
                except:
                    continue

            # If want to do WHITEBOX
            restrict_space = False
            if restrict_space:
                new_dict = copy.deepcopy(data)
                # The restrictions are
                # ['block_size_x*block_size_y>=64', 'tile_size_x*tile_size_y<30']
                temp = []
                temp2 = []
                for k in data['cache'].keys():
                    bs_x = data['cache'][k]['block_size_x']
                    bs_y = data['cache'][k]['block_size_y']
                    ts_x = data['cache'][k]['tile_size_x']
                    ts_y = data['cache'][k]['tile_size_y']
                    #print(bs_x, bs_y, ts_x, ts_y)
                    if bs_x*bs_y not in temp:
                        temp.append(bs_x*bs_y)
                    if ts_x*ts_y not in temp2:
                        temp2.append(ts_x*ts_y)
                    if not bs_x*bs_y >= 64:
                        del new_dict['cache'][k]
                        raise Exception("PAUSE")
                    if not ts_x*ts_y < 30:
                        del new_dict['cache'][k]
                        raise Exception("PAUSE")
                    if not bs_x*bs_y <= 1024:
                        del new_dict['cache'][k]

                temp.sort()
                temp2.sort()
                print(temp)
                print(temp2)
                print(len(data['cache'].keys()))
                print(len(new_dict['cache'].keys()))
                data = new_dict

            if not restrict_space:
                newfilename = filename[:-5] + '_processed' + '.json'
            else:
                newfilename = filename[:-5] + '_processed_whitebox' + '.json'
            with open(processed_data_path + newfilename, 'w') as outfile:
                json.dump(data, outfile)


if __name__ == '__main__':
    process()
