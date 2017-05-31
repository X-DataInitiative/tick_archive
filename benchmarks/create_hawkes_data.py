import os

import numpy as np
import shutil

from tick.simulation import SimuHawkesExpKernels

COEFFS_FILE = "hawkes_data/hawkes_coeffs_n_nodes_{}.txt"
SIMULATION_FILE = "hawkes_data/hawkes_simulation_n_nodes_{}_end_time_{}.txt"
TEST_COEFFS_FILE = "hawkes_data/hawkes_test_coeffs_n_nodes_{}.txt"


def create_parameters(n_nodes):
    if n_nodes == 1:
        baseline = np.array([0.4])
        adjacency = np.array([[0.6]])

    elif n_nodes == 2:
        baseline = np.array([0.25, 0.3])
        adjacency = np.array([[.3, 0.], [.6, .2]])

    elif n_nodes == 4:
        baseline = np.array([0.17, 0., 0.12, 0.09])
        adjacency = np.array([[.3, 0., 0.2, 0.1],
                              [.3, .2, 0.1, 0.],
                              [0.1, 0.2, 0., 0.3],
                              [0.2, 0., 0.1, 0.2]])
    elif n_nodes == 16:
        np.random.seed(10)
        floor = 100
        max_rand = 0.657
        baseline = np.round(
            10 * np.random.uniform(0, max_rand, n_nodes)) / floor
        adjacency = np.round(10 * np.random.uniform(0, max_rand, (n_nodes,
                                                                  n_nodes))) / floor
    else:
        raise (ValueError("Unhandled number of nodes"))

    hawkes_coeffs = np.vstack((baseline.reshape(1, n_nodes), adjacency))
    np.savetxt(COEFFS_FILE.format(n_nodes), hawkes_coeffs, fmt='%.2f')


def get_decay(n_nodes):
    if n_nodes == 1:
        return 1.5
    elif n_nodes == 2:
        return 2
    elif n_nodes == 4:
        return 0.5
    elif n_nodes == 16:
        return 1.
    else:
        raise (ValueError("Unhandled number of nodes"))


def get_parameters(n_nodes):
    decay = get_decay(n_nodes)
    coeffs = np.loadtxt(COEFFS_FILE.format(n_nodes))
    if n_nodes == 1:
        coeffs = coeffs.reshape(2, 1)
    baseline = coeffs[0]
    adjacency = coeffs[1:, :]

    return decay, baseline, adjacency


def simulate_hawkes_data(n_nodes, end_time):
    decays, baseline, adjacency = get_parameters(n_nodes)

    hawkes_exp_kernels = SimuHawkesExpKernels(
        adjacency=adjacency, decays=decays, baseline=baseline,
        end_time=end_time, seed=1039, verbose=False)

    hawkes_exp_kernels.simulate()

    with open(SIMULATION_FILE.format(n_nodes, end_time), "ba") as sim_file:
        for node in range(hawkes_exp_kernels.n_nodes):
            np.savetxt(sim_file, hawkes_exp_kernels.timestamps[node],
                       newline=' ')
            sim_file.write(b'\n')

        print("Saved simulation with {} nodes and {} total jumps"
              .format(n_nodes, hawkes_exp_kernels.n_total_jumps))


def simulate_test_coeffs(n_nodes):
    np.random.seed(3209)

    test_coeffs = np.random.rand(n_test_coeffs, n_nodes * (1 + n_nodes))
    np.savetxt(TEST_COEFFS_FILE.format(n_nodes), test_coeffs)


def get_simulation(n_nodes, end_time):
    timestamps = []
    with open(SIMULATION_FILE.format(n_nodes, end_time), "r") as sim_file:
        for node in range(n_nodes):
            timestamps.append(np.fromstring(sim_file.readline(), sep=' '))
    return timestamps


def get_test_coeffs(n_nodes):
    return np.loadtxt(TEST_COEFFS_FILE.format(n_nodes))


n_nodes_sample = [1, 2, 4, 16]
end_times = [100, 10000, 20000, 50000, 100000, 200000, 500000, 1000000]
n_test_coeffs = 100

if __name__ == "__main__":
    if os.path.exists("hawkes_data"):
        shutil.rmtree("hawkes_data")
    os.mkdir("hawkes_data")
    for n_nodes in [1, 2, 4, 16]:
        create_parameters(n_nodes)
        simulate_test_coeffs(n_nodes)

        for end_time in end_times:
            simulate_hawkes_data(n_nodes, end_time)

        print()
