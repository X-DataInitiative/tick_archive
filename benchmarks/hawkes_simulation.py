import numpy as np
import time

from tick.simulation import SimuHawkesExpKernels

end_time = 10000000
decays = np.array([[2., 2.], [2., 2.]])
baseline = np.array([0.12, 0.07])
adjacency = np.array([[.3, 0.], [.6, .21]])

hawkes_exp_kernels = SimuHawkesExpKernels(
    adjacency=adjacency, decays=decays, baseline=baseline,
    end_time=end_time, seed=1039, verbose=False)

print('baseline\n', baseline)
print('decays\n', decays)
print('adjacency\n', adjacency)
print('adjacency * decays\n', adjacency * decays)

start_time = time.clock()
hawkes_exp_kernels.simulate()
simulation_time = time.clock() - start_time

print("Simulating {} events costs {} secs"
      .format(hawkes_exp_kernels.n_total_jumps, simulation_time))
