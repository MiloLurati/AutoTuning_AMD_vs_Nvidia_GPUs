#!/usr/bin/env python
import os
import numpy as np
from collections import OrderedDict
import kernel_tuner as kt
import sys
from kernel_tuner.file_utils import store_output_file, store_metadata_file
import json

nr_dms = 2048
nr_samples = 25000
nr_channels = 1536
max_shift = 650
nr_samples_per_channel = (nr_samples+max_shift)
down_sampling = 1
dm_first = 0.0
dm_step = 0.02

channel_bandwidth = 0.1953125
sampling_time = 0.00004096
min_freq = 1425.0
max_freq = min_freq + (nr_channels-1) * channel_bandwidth



def get_shifts():
    max_freq = min_freq + ((nr_channels - 1) * channel_bandwidth)
    inverse_high_freq = 1/max_freq**2
    time_unit = nr_samples * sampling_time

    channels = np.arange(nr_channels, dtype=np.float32)
    inverse_freq = 1.0 / (min_freq + (channels * channel_bandwidth))**2
    # 4148.808 is the time delay per dispersion measure, a constant in the dispersion equation
    shifts_float = (4148.808 * (inverse_freq - inverse_high_freq) * (nr_samples / down_sampling)) / time_unit
    shifts_float[-1] = 0
    return shifts_float



def create_reference():

    input_samples = np.random.randn(nr_samples_per_channel*nr_channels).astype(np.uint8)

    output_arr = np.zeros(nr_dms*nr_samples, dtype=np.float32)
    shifts = get_shifts()

    kernel_name = "dedispersion_reference"
    kernel_string = "dedispersion.cc"
    args = [input_samples, output_arr, shifts]

    reference = kt.run_kernel("dedispersion_reference", "dedispersion.cc", 1, args, {}, lang="C")

    print(reference[1][:50])

    np.save("input_ref", input_samples, allow_pickle=False)
    np.save("shifts_ref", shifts, allow_pickle=False)
    np.save("dedisp_ref", reference[1], allow_pickle=False)

def tune(device):

    input_samples = np.load("input_ref.npy")
    output_arr = np.zeros(nr_dms*nr_samples, dtype=np.float32)
    shifts = np.load("shifts_ref.npy")

    # ensure consistency of the input files
    assert max_shift > (dm_first + nr_dms * dm_step) * shifts[0]

    args = [input_samples, output_arr, shifts]

    answer = [None, np.load("dedisp_ref.npy"), None]

    problem_size = (nr_samples, nr_dms, 1)
    tune_params = OrderedDict()
    tune_params["block_size_x"] = [1, 2, 4, 8] + [16*i for i in range(1,3)]
    tune_params["block_size_y"] = [8*i for i in range(4,33)]
    tune_params["block_size_z"] = [1]
    tune_params["tile_size_x"] = [i for i in range(1,5)]
    tune_params["tile_size_y"] = [i for i in range(1,9)]
    tune_params["tile_stride_x"] = [0, 1]
    tune_params["tile_stride_y"] = [0, 1]
    tune_params["loop_unroll_factor_channel"] = [0] #+ [i for i in range(1,nr_channels+1) if nr_channels % i == 0] #[i for i in range(nr_channels+1)]

    [print(k,v) for k,v in tune_params.items()]

    cp = [f"-I{os.path.dirname(os.path.realpath(__file__))}"]


    check_block_size = "32 <= block_size_x * block_size_y <= 1024"
    check_loop_x = "loop_unroll_factor_x <= tile_size_x and tile_size_x % loop_unroll_factor_x == 0"
    check_loop_y = "loop_unroll_factor_y <= tile_size_y and tile_size_y % loop_unroll_factor_y == 0"
    check_loop_channel = f"loop_unroll_factor_channel <= {nr_channels} and loop_unroll_factor_channel and {nr_channels} % loop_unroll_factor_channel == 0"

    check_tile_stride_x = "tile_size_x > 1 or tile_stride_x == 0"
    check_tile_stride_y = "tile_size_y > 1 or tile_stride_y == 0"

    config_valid = [check_block_size, check_tile_stride_x, check_tile_stride_y]

    metrics = OrderedDict()
    gbytes = (nr_dms * nr_samples * nr_channels)/1e9
    metrics["GB/s"] = lambda p: gbytes / (p['time'] / 1e3)

    kernel_file = "dedispersion.cu.hip"

    results, env = kt.tune_kernel("dedispersion_kernel", kernel_file, problem_size, args, tune_params,
                                  answer=answer, compiler_options=cp, restrictions=config_valid, device=0,
                                  cache=f"dedisp_{device}_cache.json", lang='HIP', iterations=32, metrics=metrics)

    env_file = open(f'dedisp_{device}_env.json', "w")
    json.dump(env, env_file, indent = 6)
    env_file.close()
    
    store_metadata_file(f'dedisp_{device}_metadata.json')

if __name__ == "__main__":

    arg1 = sys.argv[1]

    if arg1 not in ('a100', 'a4000', "w6600", "mi50", "mi250"):
        print("wrong argument")
        exit()

    if not os.path.isfile("dedisp_ref.npy"):
        print("Reference file does not exist, first creating reference output on the CPU")
        create_reference()

    print("Tuning ... ")
    tune(arg1)


