import time
from benchmarks.create_hawkes_data import get_parameters

from tick.simulation import SimuHawkesExpKernels, SimuHawkesMulti


if __name__ == "__main__":
    n_simulations = 100

    n_nodes_sample = [1, 2, 4, 16]
    end_times = [1000, 2000, 5000, 10000, 20000, 50000, 100000]
    n_threads = [1, 4, 16]

    for n_thread in n_threads:
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

                multi = SimuHawkesMulti(hawkes_exp_kernels,
                                        n_simulations=n_simulations,
                                        n_threads=n_thread)

                start_time = time.time()
                multi.simulate()
                simulation_time = time.time() - start_time

                if False:
                    print("Simulating {} events costs {} secs"
                          .format(sum(multi.n_total_jumps), simulation_time))
                else:
                    print("simulation,tick {},{},{},{:.6f}"
                          .format(n_thread, n_nodes, sum(multi.n_total_jumps),
                                  simulation_time))
