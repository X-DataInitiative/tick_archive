import os
import time

import numpy as np

from tick.simulation import SimuHawkesExpKernels
from tick.optim.model import ModelHawkesFixedExpKernLogLik

SIMULATION_FILE = "hawkes_data/hawkes_simulation.txt"
TEST_COEFFS_FILE = "hawkes_data/hawkes_test_coeffs.txt"


def simulate_hawkes_data(decay):
    """Generates benchmarks data that will be used by all tests
    """
    end_time = 100000
    baseline = np.array([0.12, 0.07])
    adjacency = np.array([[.3, 0.], [.6, .21]])
    n_test_coeffs = 1000

    hawkes_exp_kernels = SimuHawkesExpKernels(
        adjacency=adjacency, decays=decay, baseline=baseline,
        end_time=end_time, seed=1039, verbose=False)

    hawkes_exp_kernels.simulate()

    if not os.path.exists("hawkes_data"):
        os.mkdir("hawkes_data")

    with open(SIMULATION_FILE, "ba") as sim_by_node_file:
        for node in range(hawkes_exp_kernels.n_nodes):
            np.savetxt(sim_by_node_file, hawkes_exp_kernels.timestamps[node],
                       newline=' ')
            sim_by_node_file.write(b'\n')

    np.random.seed(3209)
    test_coeffs = np.random.rand(n_test_coeffs, baseline.size + adjacency.size)
    np.savetxt(TEST_COEFFS_FILE, test_coeffs)

    print("Saved simulation with {} nodes and {} total jumps\n"
          .format(n_nodes, hawkes_exp_kernels.n_total_jumps))


decay = 2.
n_nodes = 2

data_exists = os.path.exists(SIMULATION_FILE)
if not data_exists:
    simulate_hawkes_data(decay)

timestamps = []
with open(SIMULATION_FILE, "r") as sim_by_node_file:
    for node in range(n_nodes):
        timestamps.append(np.fromstring(sim_by_node_file.readline(), sep=' '))

test_coeffs = np.loadtxt(TEST_COEFFS_FILE)

model = ModelHawkesFixedExpKernLogLik(decay)
model.fit(timestamps)
n_total_jumps = sum(map(len, timestamps))

n_tries_compute_weights = 100
start_time = time.clock()
for _ in range(n_tries_compute_weights):
    model._model.compute_weights()
compute_weights_time = (time.clock() - start_time) / n_tries_compute_weights
print('Time needed to compute weights {:.6f}'.format(compute_weights_time))

start_time = time.clock()
for test_coeff in test_coeffs:
    loss = model.loss(test_coeff)
average_compute_likelihood_time = (time.clock() - start_time) / len(test_coeffs)

print('Average time to compute likelihood {:.6f}'
      .format(average_compute_likelihood_time))

print('Time needed for first likelihood {:.6f}'
      .format(compute_weights_time + average_compute_likelihood_time))

print("Negative loglikelihood value on first test coeff {:.6f}"
      .format(model.loss(test_coeffs[0]) * n_total_jumps))
