import os
import json
import time
import numpy as np
import argparse
from . import device_info as cuda
from tabulate import tabulate

os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"


dtype_dict = {
    'fp16': 'float16',
    'fp32': 'float32',
}

def environment_info(args) -> str:
    return str(tabulate(
        headers=[
            'Name', 'Value'
        ],
        tabular_data=[
            ['GPU', cuda.query_device_name()],
            ['Arch', cuda.query_arch()],
            ['Compute Capacity', cuda.query_compute_capability()],
            ['Current SM Clock (MHz)', cuda.query_gpu_current_clock()],
            ['Current Memory Clock (MHz)', cuda.query_memory_current_clock()],
            ['Warmup/Number/Repeat', '{} / {} / {}'.format(args.warmup, args.number, args.repeat)]
        ]
    ))



parser = argparse.ArgumentParser(description='auto benchmark script.')


# general parameters
parser.add_argument('--workload', type=str, choices=['gemm', 'batch-gemm', 'conv1d', 'conv2d', 'attention', 'layernorm'],
                    required=True,
                    help='The model to benchmark.')
parser.add_argument('--engine', type=str, choices=['triton', 'cutlass', 'tvm_ms', 'autotvm', 'ansor', 'tf', 'tf_xla', 'torch'], required=True,
                    help='engine to run.')
parser.add_argument('--out_dir', type=str, default='./results/',
                    help='Output directory.')
parser.add_argument('--warmup', type=int, default=10, help='Number of warmups.')
parser.add_argument('--number', type=int, default=10, help='Number of runs per repeat.')
parser.add_argument('--repeat', type=int, default=10, help='Number of repeats.')


parser.add_argument('--input_dtype', choices=['f16', 'f32'], default='f16')
parser.add_argument('--acc_dtype', choices=['f16', 'f32'], default='f16')
parser.add_argument('--out_dtype', choices=['f16', 'f32'], default='f16')

# parser.add_argument('--parallel_k', choices=['disabled', 'default', 'search', '2', '4', '6', '8'], default='default')

# tvm number of trial per task
parser.add_argument('--ansor_trial', type=int, default=800, help='Number of trial per task in autotvm , default 800.')
parser.add_argument('--autotvm_trial', type=int, default=1000, help='Number of trial per task in ansor, default 1000.')
parser.add_argument('--meta_trial', type=int, default=1000, help='Number of trial per task in meta schedule, default 1000.')



# ======

# model agnostic parameters
parser.add_argument('--bs', type=int, default=1, help='Batch size.')

