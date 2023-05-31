import tvm
from tvm import meta_schedule as ms
from utils.workloads_fp16 import create_te_workload_f16
from utils.util_config import *
from typing import List, Optional, Tuple, Union

import os
import json
import argparse
from tabulate import tabulate


from utils import cuda

# from engine_tvm_ansor import ansor_tune
# from engine_tvm_ms import ms_tune
# from engine_cutlass import search



import os
os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

WORKLOADS = [
    "C1D",
    "C2D",
    "C3D",
    "GEMM-1024-1024-1024",
    "GEMM-4096-4096-4096",
]

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
            # ['Warmup/Number/Repeat', '{} / {} / {}'.format(args.warmup, args.number, args.repeat)]
        ]
    ))



def parse_args():
    args = argparse.ArgumentParser(description='auto benchmark script.')
    args.add_argument(
        "--workload",
        type=str,
        required=True,
    )
    args.add_argument("--engine", type=str, choices=['tvm_ansor', 'tvm_ms', 'triton', 'cutlass', 'torch', 'nvcublas'], required=True)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--target", type=str)
    args.add_argument("--num_trials", type=int, default=100)
    # args.add_argument("--work_dir", type=str)
    args.add_argument("--log_dir", type=str, default="./results/")

    use_rpc = args.add_mutually_exclusive_group()
    use_rpc.add_argument("--local", action="store_false", dest="use_rpc", default=False)
    use_rpc.add_argument("--rpc", action="store_true", dest="use_rpc")
    args.add_argument("--rpc-host", type=str)
    args.add_argument("--rpc-port", type=int)
    args.add_argument("--rpc-key", type=str)
    args.add_argument("--workers", type=int)
    args.add_argument("--alloc-repeat", type=int, default=1)

    args.add_argument("--input_dtype", type=str,
                      choices=["f16", "f32"], default="f16")
    args.add_argument("--acc_dtype", type=str,
                      choices=["f16", "f32"], default="f16")
    args.add_argument("--out_dtype", type=str,
                      choices=["f16", "f32"], default="f16")
    args.add_argument("--cutlass-home", type=str)
    parsed = args.parse_args()
    parsed.cutlass_home = parsed.cutlass_home or os.getenv("CUTLASS_HOME")
    assert (
        parsed.cutlass_home
    ), "Please specify 'CUTLASS_HOME', by either setting the environment variable or using --cutlass-home"
    parsed.profiler = f"{parsed.cutlass_home}/build/tools/profiler/cutlass_profiler"
    parsed = args.parse_args()
    parsed.target = parsed.target or os.environ.get("TVM_TARGET")
    parsed.target = Target(parsed.target)
    # parsed.work_dir = parsed.work_dir or f"logs/"
    if parsed.use_rpc:
        rpc_host = parsed.rpc_host or os.environ.get("TVM_RPC_HOST")
        rpc_port = parsed.rpc_port or int(os.environ.get("TVM_RPC_PORT"))
        rpc_key = parsed.rpc_key or os.environ.get("TVM_RPC_KEY")
        rpc_config = ms.runner.RPCConfig(
            tracker_host=rpc_host,
            tracker_port=rpc_port,
            tracker_key=rpc_key,
            session_timeout_sec=60,
        )
        workers = parsed.workers or rpc_config.count_num_servers(allow_missing=False)
        parsed.runner = partial(
            ms.runner.RPCRunner, rpc_config=rpc_config, max_workers=workers
        )
    else:
        parsed.runner = ms.runner.LocalRunner
    parsed.runner = parsed.runner(
        evaluator_config=ms.runner.EvaluatorConfig(
            number=3,
            repeat=1,
            min_repeat_ms=100,
            enable_cpu_cache_flush=False,
        )
    )
    return parsed




def bench_tvm_ansor(args, out_dir):
    pass

def bench_tvm_ms(args, out_dir):
    pass

def bench_triton(args, out_dir):
    pass

def bench_cutlass(args, out_dir):
    pass

def bench_torch(args, out_dir):
    pass

def bench_nvcublas(args, out_rid):
    pass


def bench(command_line_args: Optional[str]=None):

    args = parse_args()
    print(f"current engine is {args.engine}")

    task_name = 'batch_size_{}_{}_{}_input_{}_acc_{}_output_{}'.format(args.batch_size, args.workload, args.engine, args.input_dtype, args.acc_dtype, args.out_dtype)
    print(task_name)


    bench_dict = {
        "tvm_ansor": bench_tvm_ansor,
        "tvm_ms": bench_tvm_ms,
        "triton": bench_triton,
        "cutlass": bench_cutlass,
        "torch": bench_torch,
        "cublas": bench_nvcublas,
    }
    bench_func = bench_dict[args.engine]
    out_dir = os.path.join(args.log_dir, cuda.query_device_name(short=True), 'workloads')

    if args.engine in ["tvm_ms", "tvm_ansor"]:
        trials = args.num_trials
        task_name += "_trials_{}".format(trials)
    out_dir = os.path.join(out_dir, task_name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, 'env.txt'), 'w') as f:
        f.write(environment_info(args))








if __name__ == "__main__":
    bench()