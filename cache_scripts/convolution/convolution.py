#!/usr/bin/env python
import json
import logging
import os
import sys
import time
from collections import OrderedDict

import kernel_tuner
import numpy
from kernel_tuner.util import get_config_string
from kernel_tuner.file_utils import store_output_file, store_metadata_file

unit = "GFLOP"
def ops(w, h, fw, fh):
    return (w * h * fw * fh * 2)/1e9


def tune(inputs, device):
    kernel_file = '/convolution.cu.hip'

    with open(os.path.dirname(os.path.realpath(__file__)) + kernel_file, 'r') as f:
        kernel_string = f.read()

    #setup tunable parameters
    tune_params = OrderedDict()

    image_width, image_height, filter_width, filter_height = inputs

    tune_params["block_size_x"] = [16*i for i in range(1,17)]
    tune_params["block_size_y"] = [2**i for i in range(5)]
    tune_params["tile_size_x"] = [i for i in range(1,5)]
    tune_params["tile_size_y"] = [i for i in range(1,5)]
    tune_params["read_only"] = [0,1]    #toggle using the read-only cache

    tune_params["use_padding"] = [0,1]  #toggle the insertion of padding in shared memory

    #limit the search to only use padding when its effective
    restrict = ["(use_padding==0 or (block_size_x % 32 != 0))", "((block_size_x*tile_size_x+4)*(block_size_y*tile_size_y+4) < 12*1024)"]
    restrict.append("(((block_size_x*tile_size_x+%d)*(block_size_y*tile_size_y+%d)) < 12*1024)" % (filter_width-1, filter_height-1))
    restrict.append("block_size_x * block_size_y <= 1024")

    problem_size = (image_width, image_height)
    size = numpy.prod(problem_size)
    largest_fh = filter_height
    largest_fw = filter_width
    input_size = ((problem_size[0]+largest_fw-1) * (problem_size[1]+largest_fh-1))

    output_image = numpy.zeros(size).astype(numpy.float32)
    input_image = numpy.random.randn(input_size).astype(numpy.float32)
    filter_weights = numpy.random.randn(largest_fh * largest_fw).astype(numpy.float32)

    cmem_args = {'d_filter': filter_weights}
    args = [output_image, input_image, filter_weights]

    grid_div_x = ["block_size_x", "tile_size_x"]
    grid_div_y = ["block_size_y", "tile_size_y"]

    total_flops = ops(*inputs)
    metrics = OrderedDict()
    metrics["GFLOP/s"] = lambda p: total_flops / (p["time"] / 1000.0)

    cache_BF_file = f'convolution_{device}_cache.json'

    print("BRUTE FORCE")
    # Brute Force
    results, env = kernel_tuner.tune_kernel("convolution_kernel", kernel_string,
                    problem_size, args, tune_params,
                    grid_div_y=grid_div_y, grid_div_x=grid_div_x, cmem_args=cmem_args,
                    verbose=True, metrics=metrics, lang='HIP',
                    simulation_mode=False, device=0,
                    restrictions=restrict, iterations=32, cache=cache_BF_file)
    env_file = open(f"convolution_{device}_env.json", "w")
    json.dump(env, env_file, indent = 6)
    env_file.close()
    
    # Store the metadata of this run
    store_metadata_file(f'convolution_{device}_metadata.json')
    
   
if __name__ == "__main__":

    arg1 = sys.argv[1]

    if arg1 not in ("a100", "a4000", "mi50", "w6600", "mi250"):
        print("argv[1] not valid, specify a100 or a4000 or mi50 or w6600 or mi250")
        exit()

    w = h = 4096
    fw = fh = 15
    total_flops = ops(w, h, fw, fh)

    tune([w,h,fw,fh], arg1)


