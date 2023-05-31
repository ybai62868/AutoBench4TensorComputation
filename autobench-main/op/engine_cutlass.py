import argparse
import os
import json
import subprocess
from typing import Tuple, Union


def _run_cutlass(instruction: str, workload: str, out_dir: str):
    print("Running:", workload)
    logs = subprocess.check_output(instruction, shell=True)
    logs = logs.decode("utf-8")
    logs = logs.split("\n")
    csv_index = logs.index("CSV Results:")

    csv_file = os.path.join(out_dir, f"{workload}.csv")
    with open(csv_file, "w") as f:
        f.write("\n".join(logs[csv_index + 2:]))

    max_gflops = max(
        [float(log.split(",")[-1]) for log in logs[csv_index + 3:] if log]
    )
    print(f"{workload}: {max_gflops/1024} TFLOPS")
    print(f"Full benchmark results have been written to {csv_file}")


def _run_gemm(
    workload: str,
    b: int,
    n: int,
    m: int,
    k: int,
    acc_dtype: str,
    out_dtype: str,
    profiler: str,
    out_dir: str
):
    _run_cutlass(
        f"{profiler} --operation=gemm"
        f" --batch_count={b} --n={n} --m={m} --k={k}"
        f" --A=f16:row --B=f16:column --C={out_dtype}"
        f" --accumulator-type={acc_dtype}",
        workload=f"{workload}-{b}-{acc_dtype}-{out_dtype}",
        out_dir=out_dir,
    )


def _run_conv(
    workload: str,
    n: int,
    d: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kernel_size: Union[int, Tuple[int]],
    stride: Union[int, Tuple[int]],
    padding: Union[int, Tuple[int]],
    dilation: Union[int, Tuple[int]],
    acc_dtype: str,
    out_dtype: str,
    profiler: str,
    out_dir: str,
):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size,) * 2
    if isinstance(stride, int):
        stride = (stride,) * 2
    if isinstance(padding, int):
        padding = (padding,) * 2
    if isinstance(dilation, int):
        dilation = (dilation,) * 2
    operation = "Conv2d" if d == 0 else "Conv3d"
    _run_cutlass(
        f"{profiler} --operation={operation} --Activation=f16:nhwc --Filter=f16:nhwc"
        f" --n={n} --h={h} --w={w} --c={ci} --k={co} {f'--d={d}' if d != 0 else ''}"
        f" --r={kernel_size[0]} --s={kernel_size[0]} --pad_h={padding[0]} --pad_w={padding[1]}"
        f" --stride_h={stride[0]} --stride_w={stride[1]}"
        f" --dilation_h={dilation[0]} --dilation_w={dilation[1]}"
        f" --accumulator-type={acc_dtype} --Output={out_dtype}",
        workload=f"{workload}-{n}-{acc_dtype}-{out_dtype}",
        out_dir=out_dir,
    )


def C1D(
    batch: int,
    l: int,
    ci: int,
    co: int,
    kernel: int,
    stride: int,
    padding: int,
    acc_dtype: str,
    out_dtype: str,
    profiler: str,
    out_dir: str,
):
    # _, l, ci, co, kernel, stride, padding = CONFIGS["C1D"]
    return _run_conv(
        "C1D",
        batch,
        0,  # d
        1,
        l,
        ci,
        co,
        (1, kernel),
        (1, stride),
        (0, padding),
        (1, 1),
        acc_dtype,
        out_dtype,
        profiler,
        out_dir,
    )


def C2D(
    batch: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kernel: int,
    stride: int,
    padding: int,
    acc_dtype: str,
    out_dtype: str,
    profiler: str,
    out_dir: str,
):
    # _, h, w, ci, co, kernel, stride, padding = CONFIGS["C2D"]
    return _run_conv(
        "C2D",
        batch,
        0,  # d
        h,
        w,
        ci,
        co,
        kernel,
        stride,
        padding,
        1,  # dilation
        acc_dtype,
        out_dtype,
        profiler,
        out_dir,
    )


def C3D(
    batch: int,
    d: int,
    h: int,
    w: int,
    ci: int,
    co: int,
    kernel: int,
    stride: int,
    padding: int,
    acc_dtype: str,
    out_dtype: str,
    profiler: str,
    out_dir: str,
):
    # _, d, h, w, ci, co, kernel, stride, padding = CONFIGS["C3D"]
    return _run_conv(
        "C3D",
        batch,
        d,
        h,
        w,
        ci,
        co,
        kernel,
        stride,
        padding,
        1,  # dilation
        acc_dtype,
        out_dtype,
        profiler,
        out_dir,
    )


def DIL(batch: int, acc_dtype: str, out_dtype: str):
    _, h, w, ci, co, kernel, stride, padding, dilation = CONFIGS["DIL"]
    return _run_conv(
        "DIL",
        batch,
        0,  # d
        h,
        w,
        ci,
        co,
        kernel,
        stride,
        padding,
        dilation,  # dilation
        acc_dtype,
        out_dtype,
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
    out_dir: str,
):
    return _run_gemm(
        workload,
        batch,
        m,
        n,
        k,
        acc_dtype,
        out_dtype,
        profiler,
        out_dir,
    )




def bench_cutlass(args, out_dir):
    cutlass_workload_dict = {
        "GEMM-1024-1024-1024": [1024, 1024, 1024],
        "GEMM-4096-4096-4096": [4096, 4096, 4096],
        "C1D": [256, 64, 64, 3, 1, 1],
        "C2D": [56, 56, 64, 64, 3, 1, 1],
        "C3D": [16, 56, 56, 64, 64, 3, 1, 1],
    }

    cutlass_profiler = f"{args.cutlass_home}/build/tools/profiler/cutlass_profiler"
    if args.workload[0:4] == "GEMM":
        GEMM(workload=args.workload,
            batch=args.batch_size, 
            m=cutlass_workload_dict[args.workload][0],
            n=cutlass_workload_dict[args.workload][0],
            k=cutlass_workload_dict[args.workload][0],
            acc_dtype=args.acc_dtype, 
            out_dtype=args.out_dtype, 
            profiler=cutlass_profiler,
            out_dir=out_dir,
            )
    elif args.workload[0] == "C" and args.workload[1] == "1":
        C1D()
    elif args.workload[0] == "C" and args.workload[1] == "2":
        C2D()
    elif args.workload[0] == "C" and args.workload[1] == "3":
        C3D()
    else:
        raise Exception("Unsupported operator!")