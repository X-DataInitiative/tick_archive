import time
from benchmarks.create_hawkes_data import get_parameters

from tick.simulation import SimuHawkesExpKernels

if __name__ == "__main__":
    n_nodes_sample = [1, 2, 4, 16]
    end_times = [100000]#, 200000, 500000, 1000000, 2000000, 5000000, 10000000]

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

            start_time = time.clock()
            hawkes_exp_kernels.simulate()
            simulation_time = time.clock() - start_time

            if False:
                print("Simulating {} events costs {} secs"
                      .format(hawkes_exp_kernels.n_total_jumps, simulation_time))
            else:
                print("simulation,tick,{},{},{:.6f}"
                      .format(n_nodes, hawkes_exp_kernels.n_total_jumps,
                              simulation_time))
