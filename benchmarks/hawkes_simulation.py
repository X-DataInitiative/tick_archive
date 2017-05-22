import numpy as np
import time

from tick.simulation import SimuHawkesExpKernels


def get_parameters(n_nodes):
    if n_nodes == 1:
        decays = np.array([[1.5]])
        baseline = np.array([0.4])
        adjacency = np.array([[0.6]])

    elif n_nodes == 2:
        decays = np.array([[2., 2.], [2., 2.]])
        baseline = np.array([0.25, 0.3])
        adjacency = np.array([[.3, 0.], [.6, .2]])

    elif n_nodes == 4:
        decays = 0.5
        baseline = np.array([0.17, 0., 0.12, 0.09])
        adjacency = np.array([[.3, 0., 0.2, 0.1],
                              [.3, .2, 0.1, 0.],
                              [0.1, 0.2, 0., 0.3],
                              [0.2, 0., 0.1, 0.2]])
    else:
        raise (ValueError("Unhandled number of nodes"))

    return decays, baseline, adjacency


if __name__ == "main":

    n_nodes_sample = [1, 2, 4]
    end_times = [100000, 1000000, 10000000]

    for n_nodes in n_nodes_sample:
        decays, baseline, adjacency = get_parameters(n_nodes)

        print("\nn nodes", n_nodes)
        print('baseline\n', baseline)
        print('decays\n', decays)
        print('adjacency\n', adjacency)
        print('adjacency * decays\n', adjacency * decays)

        for end_time in end_times:
            hawkes_exp_kernels = SimuHawkesExpKernels(
                adjacency=adjacency, decays=decays, baseline=baseline,
                end_time=end_time, seed=1039, verbose=False)

            start_time = time.clock()
            hawkes_exp_kernels.simulate()
            simulation_time = time.clock() - start_time

            print("Simulating {} events costs {} secs"
                  .format(hawkes_exp_kernels.n_total_jumps, simulation_time))
