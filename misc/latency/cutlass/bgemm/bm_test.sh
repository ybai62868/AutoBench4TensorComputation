#### cover the cutlass/examples/14_ampere_tf32_tensorop_gemm/ampere_tf32_tensorop_gemm.cu with the file in this folder, and then make again!
#### if you want to test the batch gemm, change the line of 105 (batch_count = 12) and then make again



# ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=512 --n=512 --k=64
# ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=512 --n=64 --k=512
# ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=1024 --k=64
./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=64 --k=1024
