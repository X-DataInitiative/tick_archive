# License: BSD 3 clause

import numpy as np
from tick.simulation.build.simulation import Hawkes_custom
from tick.simulation import HawkesKernel0, HawkesKernelExp, HawkesKernelPowerLaw, \
    HawkesKernelSumExp
from tick.simulation import SimuHawkes

n_nodes = 2
dim = n_nodes
seed = 3007
MaxN_of_f = 10
f_i = [np.array([1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7]), np.array([1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])]

end_time = 100000
betas = np.array([10.0, 100, 500, 5000])

U = len(betas)
kernels = np.array([
            [HawkesKernelSumExp(np.array([0.2, 0.15, 0.1, 0.1]), betas), HawkesKernelSumExp(np.array([0.3, 0, 0.1, 0.1]), betas)],
            [HawkesKernelSumExp(np.array([0., 0.2, 0.2, 0.0]), betas), HawkesKernelSumExp(np.array([0., 0.4, 0, 0.1]), betas)]
        ])

simu_model = SimuHawkes(kernels=kernels, end_time=end_time, custom=True, seed=seed, MaxN_of_f = MaxN_of_f, f_i=f_i)
for i in range(n_nodes):
    simu_model.set_baseline(i, 0.4 + 0.1 * i)
    for j in range(n_nodes):
        simu_model.set_kernel(i, j, kernels[i, j])
simu_model.track_intensity(0.1)
# print(simu_model.simulate)
simu_model.simulate()
##################################################################################################################





##################################################################################################################
from tick.optim.model import ModelHawkesSumExpCustom
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1

timestamps = simu_model.timestamps
timestamps.append(np.array([]))
global_n = np.array(simu_model._pp.get_global_n())
global_n = np.insert(global_n, 0, 0).astype(int)

model = ModelHawkesSumExpCustom(betas, MaxN_of_f, n_threads=8)
model.fit(timestamps, global_n, end_time)
#############################################################################
prox = ProxZero()

# solver = AGD(step=5e-2, linesearch=False, max_iter= 350)
solver = AGD(step=1e-3, linesearch=False, max_iter=5000, print_every=50)
solver.set_model(model).set_prox(prox)

x_real = np.array(
    [0.4, 0.5,   0.2, 0.3, 0, 0,  0.15, 0, 0.2, 0.4,  0.1, 0.1, 0.2, 0, 0.1, 0.1, 0, 0.1,
     1., 0.7, 0.8, 0.6, 0.5, 0.8, 0.3, 0.6, 0.2, 0.7,  1., 0.6, 0.8, 0.8, 0.6, 0.6, 0.5, 0.8, 0.3, 0.6])
x0 = np.array(
    [0.6, 0.8,   0.2, 0.2, 0.2, 0.2,  0.2, 0.2, 0.2, 0.2,  0.4, 0.4, 0.4, 0.4, 0.5, 0.5, 0.5, 0.5,
     0.5, 0.5, 0.1, 0.9, 0.9, 0.3, 0.6, 0.8, 0.5,  0.5, 0.5, 0.9, 0.9, 0.3, 0.6, 0.7, 0.8, 0.5, 0.5, 0.5])
solver.solve(x0)

print(model.loss(x_real))
print(model.loss(solver.solution))



# normalisation the fitting result as f_i(0) = 1
coeff = solver.solution
Total_States = 10

for i in range(dim):
    fi0 = coeff[dim + dim * dim * U + i * Total_States]
    coeff[i] *= fi0
    for u in range(U):
        coeff[dim + dim * dim * u + i * dim: dim + dim * dim * u + (i + 1) * dim] *= fi0
    coeff[dim + dim * dim * U + i * Total_States: dim + dim * dim * U + (i + 1) * Total_States] /= fi0


print(coeff[:2])
for i in range(U):
    print(coeff[2 + 4 * i : 2 + 4 * (i+1)])
print(coeff[-2 * Total_States:-Total_States])
print(coeff[-Total_States:])