import os
from subprocess import PIPE, run

import datetime
from benchmark_util import executables, get_filename

output_dir = 'results/{:%Y_%m_%d_%H_%M_%S}'.format(datetime.datetime.now())

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

use_existing_data = False
if not use_existing_data:
    for ex in executables:
        with open(os.path.join(output_dir, get_filename(ex)), 'w') as f:
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

print("Output in %s" % output_dir)
