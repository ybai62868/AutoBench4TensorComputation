import argparse
from functools import partial
import os
import subprocess
from typing import Tuple, Union


def bench_cutlass(instruction: str, workload: str):
    print("++++++++++++++++++++++++++++++++++++++++++")
    logs = subprocess.check_output(instruction, shell=True)
    logs = logs.decode('utf-8')
    logs = logs.split("\n")
    csv_index = logs.index("CSV Results:")

    csv_file = os.path.join(args.log_dir, f"{workload}.csv")
    with open(csv_file, "w") as f:
        f.write("\n".join(logs[csv_index + 2:]))

    max_gflops = max(
        [float(log.split(",")[-1]) for log in logs[csv_index + 3 :] if log]
    )
    print(f"{workload}: {max_gflops} GFLOPS")
    print(f"benchmark results have been written to {csv_file}")


def run_gemm(
    workload: str,
    b: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
    profiler: str,
    log_dir: str
):
    bench_cutlass(
        f" --profiler={profiler} --operation=gemm"
        f" --batch_count={b} --n={n} --m={m} --k={k}"
        f" --A=f16:row --B=f16:column --C={out_dtype}"
        f" --accumulator-type={acc_dtype}",
        f" --log_dir={log_dir}",
        workload=f"{workload}-{b}-{acc_dtype}-{out_dtype}",
    )


def GEMM(
    workload: str,
    batch: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
    profiler: str,
    log_dir: str
):
    return run_gemm(
        workload,
        batch,
        n,
        m,
        k,
        acc_dtype,
        out_dtype,
        profiler,
        log_dir,
    )
