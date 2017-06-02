import time

import numpy as np

from benchmarks.create_hawkes_data import get_test_coeffs, get_decay, \
    get_simulation, get_parameters

from tick.inference import HawkesExpKern

n_nodes_sample = [1, 2, 4, 16]
end_times = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]
n_threads = [1, 4, 16]

for n_thread in n_threads:
    for n_nodes in n_nodes_sample:

        decays = get_decay(n_nodes)

        for end_time in end_times:
            timestamps = get_simulation(n_nodes, end_time)

            n_events = sum(map(len, timestamps))

            start_time = time.time()
            learner = HawkesExpKern(decays=decays, max_iter=1000, tol=1e-7,
                                    n_threads=n_thread)
            learner.fit(timestamps, start=np.ones(
                n_nodes * (1 + n_nodes)) * 1e-2)

            fit_time = time.time() - start_time

            # end_time 100 is used for debug
            if end_time != 100:
                print("fit,tick {},{},{},{:.6f}"
                      .format(n_thread, n_nodes, n_events, fit_time))
                _, baseline, adjacency = get_parameters(n_nodes)
                dsitance_to_coeffs = np.linalg.norm(
                    adjacency - learner.adjacency) / np.prod(
                    adjacency.shape)
                if dsitance_to_coeffs > 0.1:
                    print(" |-> Did not converge", dsitance_to_coeffs)
                else:
                    pass
                    # print("Found coeffs coeff {}".format(learner.coeffs))
