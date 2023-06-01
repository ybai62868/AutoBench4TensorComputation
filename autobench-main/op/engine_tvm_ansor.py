import tvm
from typing import Optional
from tvm import tir
from tvm import meta_schedule as ms
from utils.workloads_fp16 import create_te_workload_f16

from utils.util_config import *

def bench_tvm_ansor(args, out_dir):
    if args.out_dtype == "f16":
        args.out_dtype = "float16"
    elif args.out_dtype == "f32":
        args.out_dtype = "float32"
    else:
        raise Exception("Unsupported dtype")
    mod = create_te_workload_f16(
        args.workload, batch_size=args.batch_size, out_dtype=args.out_dtype
    )
    print("start tuning with meta schedule ...")
    sch = ms.tune_tir(
        mod=mod,
        target=args.target,
        config=get_search_config(args.num_trials, args.num_trials),
        work_dir=out_dir,
        runner=args.runner,
    )

    if sch is None:
        print("No valid schedule found!")
        exit()
    print(sch.mod.script())
    print(sch.trace)




def cuda_build(mod, target, _params):
    from tvm.driver import build as tvm_build

    with tvm.transform.PassContext(config={"tir.predicate_opt": True}):
        return tvm_build(mod, target=target)


def bench_tvm_ms(args, out_dir):
    if args.out_dtype == "f16":
        args.out_dtype = "float16"
        from tvm.meta_schedule.testing import tir_tensor_intrin_fp16
    elif args.out_dtype == "f32":
        args.out_dtype = "float32"
        from tvm.meta_schedule.testing import tir_tensor_intrin
    else:
        raise Exception("Unsupported dtype")
    mod = create_te_workload_f16(
        args.workload, batch_size=args.batch_size, out_dtype=args.out_dtype
    )
    print("start tuning with meta schedule ...")
    sch = ms.tune_tir(
        mod=mod,
        target=args.target,
        config=get_search_config(args.num_trials, args.num_trials),
        work_dir=out_dir,
        builder=ms.builder.LocalBuilder(f_build=cuda_build),
        runner=args.runner,
        sch_rules=sch_rules_tensor_core,
        postprocs=postprocs_tensor_core,
    )

    if sch is None:
        print("No valid schedule found!")
        exit()
    print(sch.mod.script())
    print(sch.trace)
