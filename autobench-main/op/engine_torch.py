
import torch
import triton



def benchmark(workload:str, batch: int, n: int, m: int, k: int, acc_dtype: str, out_dtype: str, provider: str, out_dir: str):
    if acc_dtype == "f16":
        input_dtype = "torch.float16"
    a = torch.randn((batch, m, k), device='cuda', dtype=input_dtype)
    b = torch.randn((batch, k, n), device='cuda', dtype=input_dtype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), cnt = m, quantiles=quantiles)
    perf = lambda ms: batch * 2 * m * n * k  / (ms * 1e-3) / (1024 * 1024 * 1024)
    return perf(ms), perf(max_ms), perf(min_ms)


def bench_triton(args, out_dir):
    torch_workload_dict = {
        "GEMM-1024-1024-1024": [1024, 1024, 1024],
        "GEMM-4096-4096-4096": [4096, 4096, 4096],
        "C1D": [256, 64, 64, 3, 1, 1],
        "C2D": [56, 56, 64, 64, 3, 1, 1],
        "C3D": [16, 56, 56, 64, 64, 3, 1, 1],
    }
    if args.workload[0:4] == "GEMM":
        benchmark(workload=args.workload,
            batch=args.batch_size, 
            m=torch_workload_dict[args.workload][0],
            n=torch_workload_dict[args.workload][0],
            k=torch_workload_dict[args.workload][0],
            acc_dtype=args.acc_dtype, 
            out_dtype=args.out_dtype,
            provider="torch", 
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
    benchmark.run(show_plots=False, print_data=True, save_path=out_dir)



def bench_torch(args, out_dir):
    pass