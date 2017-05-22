import os
import time

import numpy as np

from benchmarks.hawkes_simulation import get_parameters
from tick.simulation import SimuHawkesExpKernels
from tick.optim.model import ModelHawkesFixedExpKernLogLik

SIMULATION_FILE = "hawkes_data/hawkes_simulation_n_nodes_{}_end_time_{}.txt"
TEST_COEFFS_FILE = "hawkes_data/hawkes_test_coeffs_n_nodes_{}.txt"


def simulate_hawkes_data(n_nodes, end_time):
    """Generates benchmarks data that will be used by all tests
    """
    decays, baseline, adjacency = get_parameters(n_nodes)

    hawkes_exp_kernels = SimuHawkesExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, seed=1039, verbose=False)

    hawkes_exp_kernels.simulate()

    if not os.path.exists("hawkes_data"):
        os.mkdir("hawkes_data")

    with open(SIMULATION_FILE.format(n_nodes, end_time),
              "ba") as sim_by_node_file:
        for node in range(hawkes_exp_kernels.n_nodes):
            np.savetxt(sim_by_node_file, hawkes_exp_kernels.timestamps[node],
                       newline=' ')
            sim_by_node_file.write(b'\n')

    # print("Saved simulation with {} nodes and {} total jumps\n"
    #       .format(n_nodes, hawkes_exp_kernels.n_total_jumps))


def simulate_test_coeffs(n_nodes):
    np.random.seed(3209)

    n_test_coeffs = 10
    test_coeffs = np.random.rand(n_test_coeffs, n_nodes * (1 + n_nodes))
    np.savetxt(TEST_COEFFS_FILE.format(n_nodes), test_coeffs)


n_nodes_sample = [1, 2, 4]
end_times = [10000, 20000, 50000, 100000, 200000, 500000, 1000000]

for n_nodes in n_nodes_sample:
    simulate_test_coeffs(n_nodes)
    test_coeffs = np.loadtxt(TEST_COEFFS_FILE.format(n_nodes))
    decays, _, _ = get_parameters(n_nodes)

    for end_time in end_times:
        data_exists = os.path.exists(SIMULATION_FILE.format(n_nodes, end_time))
        if not data_exists:
            simulate_hawkes_data(n_nodes, end_time)

        timestamps = []
        with open(SIMULATION_FILE.format(n_nodes, end_time),
                  "r") as sim_by_node_file:
            for node in range(n_nodes):
                timestamps.append(np.fromstring(sim_by_node_file.readline(),
                                                sep=' '))

        n_events = sum(map(len, timestamps))

        n_tries_first_likelihood = 10
        start_time = time.clock()
        for _ in range(n_tries_first_likelihood):
            model = ModelHawkesFixedExpKernLogLik(np.mean(decays))
            model.fit(timestamps)
            model.loss(test_coeffs[0])
        first_likelihood_time = (time.clock() - start_time) / n_tries_first_likelihood

        start_time = time.clock()
        for test_coeff in test_coeffs:
            loss = model.loss(test_coeff)
        average_compute_likelihood_time = (time.clock() - start_time) / len(test_coeffs)

        if False:
            print('Time needed for first likelihood {:.6f}'
                  .format(first_likelihood_time))

            print('Average time to compute likelihood {:.6f}'
                  .format(average_compute_likelihood_time))

            print("Negative loglikelihood value on first test coeff {:.6f}"
                  .format(model.loss(test_coeffs[0]) * n_events))

        else:
            print("likelihood,tick,{},{},{:.6f}"
                  .format(n_nodes, n_events, average_compute_likelihood_time))
    
            print("first likelihood,tick,{},{},{:.6f}"
                  .format(n_nodes, n_events, first_likelihood_time))
