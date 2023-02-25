import os
import argparse
from utils import cuda_info
from tests.torch_model.get_model import model_info
from typing import List, Optional, Tuple, Union
# from utils import Tensor
import torch

os.environ["PATH"] = os.environ["PATH"]+":/usr/local/cuda/bin/"

class BenchResult:
    def __init__(self, latencies: List[float] = None, outputs: List[torch.Tensor] = None, configs: str = None):
        self.latencies = latencies
        self.outputs: Optional[List[torch.Tensor]] = outputs
        self.configs = configs


short2long_dict = {
    'f16': 'float16',
    'f32': 'float32',
    'bf16': 'bfloat16'
}

long2short_dict = {
	'float16': 'fp16',
	'float32': 'fp32',
	'bfloat16': 'bf16'
}


def bench_torch(args, out_dir) -> BenchResult:
	result = BenchResult()
	model, input_dict = model_info(args.model, batch_size=args.batch)

	def run_func():
		model(**input_dict)

	result.latencies = benchmark_run(run_func, warmup=args.warmup, number=args.number, repeat=args.repeat)
	result.configs = "fp32"
	result.outputs = None
	return result





def engine_launch(command_line_args: Optional[str]=None):
	args = parser.parse_args()
	out_dir = args.out

	# if backend is gpu
	gpu_device = cuda.query_device_name(short=True)

	os.makedirs(out_dir, exist_ok=True)

	# bench_launch
	bench_dict = {
		"torch": bench_torch,
	}
	bench_func = bench_dict[args.exec]
	result: BenchResult = bench_func(args, out_dir)
	print(result)



if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="auto bench for deep learning frameworks and compilers")
	parser.add_argument("--model", type=str, required=True, help="model name")
	parser.add_argument("--exec", type=str, choices=["torch", "trt", "onnx", "tvm", "autotvm", "ansor", "tf", "tf_xla"],
						required=True, help="the exec engine")
	parser.add_argument("--out", type=str, required=False,
						default="./results/", help="the output of the benchmark")
	parser.add_argument("--warmup", type=str, default=20, help="warmup")
	parser.add_argument("--number", type=int, default=20, help="number of the iterations per repeat")
	parser.add_argument("--repeat", type=int, default=20, help="number of the repeats")
	parser.add_argument("--batch", type=int, default=1, help="batch size")

