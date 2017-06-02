import time
from benchmarks.create_hawkes_data import get_parameters

from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti


if __name__ == "__main__":
    n_simulations = 100

    n_nodes_sample = [1, 2, 4, 16]
    end_times = [1000, 2000, 5000, 10000, 20000, 50000, 100000]

    for n_nodes in n_nodes_sample:
        decays, baseline, adjacency = get_parameters(n_nodes)

        if False:
            print("\nn nodes", n_nodes)
            print('baseline\n', baseline)
            print('decays\n', decays)
            print('adjacency\n', adjacency)
            print('adjacency * decays\n', adjacency * decays)

        for end_time in end_times:
            hawkes_exp_kernels = SimuHawkesExpKernels(
                adjacency=adjacency, decays=decays, baseline=baseline,
                end_time=end_time, seed=1039, verbose=False)

            start_time = time.time()
            n_events = 0
            for _ in range(n_simulations):
                hawkes_exp_kernels.reset()
                hawkes_exp_kernels.simulate()
                n_events += hawkes_exp_kernels.n_total_jumps
            simulation_time = time.time() - start_time

            if False:
                print("Simulating {} events costs {} secs"
                      .format(n_events, simulation_time))
            else:
                print("simulation,tick 1,{},{},{:.6f}"
                      .format(n_nodes, n_events, simulation_time))
