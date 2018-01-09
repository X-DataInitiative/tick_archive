# License: BSD 3 clause

from tick.optim.model import ModelHawkesCustom

import numpy as np
from tick.simulation import SimuLogReg, weights_sparse_gauss, SimuHawkesExpKernels
from tick.optim.solver import GD, AGD, SGD, SVRG, SDCA
from tick.optim.prox import ProxElasticNet, ProxL2Sq, ProxZero, ProxL1
from tick.plot import plot_history

'''
pre_set parameters
'''
beta = 2.0
end_time = 10000
dim = 2

'''
generating a hawkes expnential process
'''


def get_train_data(n_nodes, betas):
    np.random.seed(263)
    baseline = np.random.rand(n_nodes)
    adjacency = np.random.rand(n_nodes, n_nodes)
    if isinstance(betas, (int, float)):
        betas = np.ones((n_nodes, n_nodes)) * betas

    sim = SimuHawkesExpKernels(adjacency=adjacency, decays=betas,
                               baseline=baseline, verbose=False,
                               seed=13487, end_time=end_time)
    sim.adjust_spectral_radius(0.8)
    adjacency = sim.adjacency
    sim.simulate()

    return sim.timestamps, baseline, adjacency


timestamps, baseline, adjacency = get_train_data(n_nodes=dim, betas=beta)

print('data size =', len(timestamps[0]), ',', len(timestamps[1]))

print(baseline)
print(adjacency)

from tick.inference import HawkesExpKern

decays = np.ones((dim, dim)) * beta
learner = HawkesExpKern(decays, penalty='l1', C=100)
learner.fit(timestamps)

print('#' * 40)
print(learner.baseline)
print(learner.adjacency)

print('#' * 40)
'''
calculate global_n and maxN_of_f
'''
MaxN_of_f = 10
global_n = np.random.randint(0, MaxN_of_f, size=1 + len(timestamps[0]) + len(timestamps[1]))
'''
create a model_custom
'''
model = ModelHawkesCustom(beta, MaxN_of_f)
model.fit(timestamps, global_n, end_time)
#############################################################################
prox = ProxL1(0.00001, positive=True)

# solver = AGD(step=5e-2, linesearch=False, max_iter= 350)
solver = AGD(step=1e-2, linesearch=False, max_iter=1500)
solver.set_model(model).set_prox(prox)

x0_3 = np.array([1,2,3,1,2,3.0,   0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,  0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
solver.solve(x0_3)

# normalisation
MaxN_of_f = MaxN_of_f - 1
solution_adj = solver.solution
#############################################################################

solution_real = np.array([0.57328379,0.75786562,0.41191205,0.37875302, 0.66579896,  0.15021593, 1,1,1,1,1,1,1,1,1,   1,1,1,1,1,1,1,1,1])

print(model.loss(np.ones(2 + 2*2 + 2 * 9)))
print(model.loss(solution_real))
print(model.loss(solution_adj))

out1= np.zeros(2 + 2*2 + 2 * 9)
model.grad(solution_real, out1)
print(out1)
out2= np.zeros(2 + 2*2 + 2 * 9)
model.grad(solution_adj, out2)
print(out2)

print(out1/out2)

for i in range(dim):
    solution_adj[i] *= solver.solution[dim + dim * dim + MaxN_of_f * i]
    solution_adj[(dim + dim * i): (dim + dim * (i + 1))] *= solver.solution[dim + dim * dim + MaxN_of_f * i]
    solution_adj[(dim + dim * dim + MaxN_of_f * i): (dim + dim * dim + MaxN_of_f * (i + 1))] /= solver.solution[
        dim + dim * dim + MaxN_of_f * i]
print(solution_adj)