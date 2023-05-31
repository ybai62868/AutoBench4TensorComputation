from typing import Optional

from tvm import tir
from tvm import meta_schedule as ms
from utils.workloads_fp16 import create_te_workload_f16

from utils.util_config import *

WORKLOADS = [
    "C1D",
    "C2D",
    "C3D",
    "GMM-1024",
    "GMM-4096",
]

ARGS = parse_args(WORKLOADS, 2000)


def ansor_tune(workload, batch_size):
    sch: Optional[tir.Schedule] = ms.tune_tir(
        mod=create_te_workload_f16(workload, batch_size, ARGS.out_dtype),
        target=ARGS.target,
        config=get_search_config(ARGS.num_trials, ARGS.num_trials),
        runner=ARGS.runner,
        work_dir=f"{ARGS.work_dir}/TVM/{workload}-{batch_size}/{ARGS.out_dtype}/",
    )
    if sch is None:
        print("No valid schedule found!")
    else:
        print(sch.mod.script())
        print(sch.trace)


if __name__ == "__main__":
    for batch_size in ARGS.batch_size:
        tune(ARGS.workload, batch_size)