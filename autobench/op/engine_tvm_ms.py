# python benchmark/end2end/tune_auto_tir_relay.py -w resnet_50 -bs 1 -n 10000

import argparse
import tvm
import os
import json
from tvm import meta_schedule as ms
from workload_fp16 import create_te_workload_f16
from functools import partial


def load_config():
    cur_path = os.path.dirname(__file__)
    config_path = os.path.join(cur_path, "configs")
    with open(config_path) as f:
        return json.load(f)
    

WORKLOADS = [
    "C1D",
    "C2D",
    "C3D",
    "DIL",
    "DEP",
    "GRP",
    "T2D",
    "CBR",
    "GMM-1024-1024-1024",
    "GMM-4096",
]
def parse_args(workload_candidates, default_trials=20000):
    args = argparse.ArgumentParser()
    args.add_argument(
        "-w",
        "--workload",
        nargs="+",
        type=str,
        choices=workload_candidates,
        required=True,
    )
    args.add_argument("-bs", "--batch-size", nargs="+", type=int, default=[1])
    args.add_argument("-t", "--target", type=str)
    args.add_argument("-n", "--num-trials", type=int, default=default_trials)
    args.add_argument("--work-dir", type=str)
    use_rpc = args.add_mutually_exclusive_group()
    use_rpc.add_argument("--local", action="store_false", dest="use_rpc", default=False)
    use_rpc.add_argument("--rpc", action="store_true", dest="use_rpc")
    args.add_argument("--rpc-host", type=str)
    args.add_argument("--rpc-port", type=int)
    args.add_argument("--rpc-key", type=str)
    args.add_argument("--workers", type=int)
    args.add_argument("--alloc-repeat", type=int, default=1)
    args.add_argument("--out-dtype", type=str, default="float16")

    parsed = args.parse_args()
    parsed.target = parsed.target or os.environ.get("TVM_TARGET")
    parsed.target = Target(parsed.target)
    parsed.work_dir = parsed.work_dir or f"logs/"
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


ARGS = parse_args(WORKLOADS, default_trials=1000)

if ARGS.out_dtype == "float16":
    from tvm.meta_schedule.testing import tir_tensor_intrin_fp16
elif ARGS.out_dtype == "float32":
    from tvm.meta_schedule.testing import tir_tensor_intrin
else:
    raise Exception("Unsupported dtype")


def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(config={"tir.predicate_opt": True}):
        return tvm_build(mod, target=target)


def tune(workload, batch_size=1):
    mod = create_te_workload_f16(
        workload, batch_size=batch_size, out_dtype=ARGS.out_dtype
    )
    sch = ms.tune_tir(
        mod=mod,
        target=ARGS.target,
        config=get_search_config(ARGS.num_trials, ARGS.num_trials),
        work_dir=f"{ARGS.work_dir}/TIR/{workload}-{batch_size}/{ARGS.out_dtype}",
        builder=ms.builder.LocalBuilder(f_build=cuda_build),
        runner=ARGS.runner,  # type: ignore
        sch_rules=sch_rules_tensor_core,
        postprocs=postprocs_tensor_core,
    )

    if sch is None:
        print("No valid schedule found!")
        exit()

    print(sch.mod.script())
    print(sch.trace)


if __name__ == "__main__":
    for workload in ARGS.workload:
        for batch in ARGS.batch_size:
            tune(workload, batch)