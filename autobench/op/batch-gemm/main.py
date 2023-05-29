from autobench import bench 


def main():
    for exe in [
        '--engine triton',
    ]:
        for wk in [
            '--workload gemm',
            '--workload batch-gemm',
            '--workload conv1d',
            '--workload conv2d',
            '--workload attention', 
        ]:
            bench('{} {}'.format(exe, wk))


if __name__ == '__main__':
    main()