executables = [
    'build_noopt/benchmark/benchmark_hawkes',
    'build_mkl/benchmark/benchmark_hawkes',
    'build_omp/benchmark/benchmark_hawkes',
    'build_omp_mkl/benchmark/benchmark_hawkes',
]


def get_filename(ex):
    return ex.replace('/', '_') + ".dat"
