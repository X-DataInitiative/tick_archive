from subprocess import PIPE, run

executables = [
    'tick/build_noopt/benchmark/benchmark_test',
    'tick/build_mkl/benchmark/benchmark_test',
    'tick/build_omp/benchmark/benchmark_test',
    'tick/build_omp_mkl/benchmark/benchmark_test',
]


def get_fn(ex):
    return ex.replace('/', '_') + ".dat"


for fn in [get_fn(ex) for ex in executables]:
    with open(fn, 'w'): pass

use_existing_data = False
if not use_existing_data:
    for ex in executables:
        with open(get_fn(ex), 'a') as f:
            print("Writing to", f)

            for num_threads in (1, 2, 4, 8):  # 16, 24, 32, 40, 48, 56, 64):
                command = [ex, str(num_threads)]
                result = run(command, stdout=PIPE, stderr=PIPE,
                             universal_newlines=True)

                f.write(result.stdout)
                print(ex, num_threads)
                print(result.stderr)

            f.flush()
        print(ex, "Done")
