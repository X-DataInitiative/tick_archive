
# coding: utf-8

# In[1]:

import sys

sys.path.append('/Users/stephane.gaiffas/Code/tick')


import numpy as np

from numpy.random import multivariate_normal, randn
from scipy.linalg.special_matrices import toeplitz
from scipy.sparse import csr_matrix


def simu_linreg(x, n, interc=None, std=1., corr=0.5, p_nnz=0.3):
    """
    Simulation of the least-squares problem
    
    Parameters
    ----------
    x : np.ndarray, shape=(d,)
        The coefficients of the model
    
    n : int
        Sample size
    
    std : float, default=1.
        Standard-deviation of the noise

    corr : float, default=0.5
        Correlation of the features matrix
    """    
    d = x.shape[0]
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)
    A *= np.random.binomial(1, p_nnz, size=A.shape)
    idx = np.nonzero(A.sum(axis=1))
    A = A[idx]
    n = A.shape[0]
    noise = std * randn(n)
    b = A.dot(x_truth) + noise
    if interc:
        b += interc
    return A, b

n = 100000
d = 30

idx = np.arange(1, d+1)

# Ground truth coefficients of the model
x_truth = (-1) ** (idx - 1) * np.exp(-idx / 10.)

A_dense, b = simu_linreg(x_truth, n, interc=-1., 
                         std=1., corr=0.8, p_nnz=0.05)
A_spars = csr_matrix(A_dense)

n, d = A_spars.shape


# In[50]:

from tick.optim.model import ModelLinReg, ModelLogReg
from tick.optim.solver import SVRG
from tick.optim.prox import ProxL2Sq, ProxZero, ProxL1, ProxTV, ProxEquality, \
    ProxElasticNet, ProxPositive, ProxGroupL1, ProxSlope

from time import time

t1 = time()


model_spars = ModelLinReg().fit(A_spars, b)
model_dense = ModelLinReg().fit(A_dense, b)

proxs = [
    ProxL2Sq(strength=1e-2, range=(0, d)),
    ProxL1(strength=1e-2, range=(0, d)),
    ProxTV(strength=1e-2, range=(0, d)),
    ProxElasticNet(strength=1e-2, ratio=0.9, range=(0, d)),
    ProxSlope(strength=1e-3, fdr=0.6, range=(0, d)),
    ProxGroupL1(strength=1e-2, blocks_start = [0, 5, 10],
                blocks_length=[5, 5, 8], range=(0, d)),
    ProxEquality(range=(0, d))
]

step = 1 / model_spars.get_lip_max()

# variance_reductions = ['last', 'rand', 'avg']
variance_reductions = ['last']
# rand_types = ['unif', 'perm']
rand_types = ['unif']
delayed_updates = ['exact', 'proba']


seed = 123
solvers_dense = []
solvers_sparse = []
solutions = []
titles = []
tol = 0.
max_iter = 30


from itertools import product

for variance_reduction, rand_type, delayed_update, prox \
        in product(variance_reductions, rand_types, delayed_updates, proxs):
    if variance_reduction == 'avg':
        continue
    solver = SVRG(step=step, tol=tol, max_iter=max_iter, verbose=False,
                  print_every=10,
                  variance_reduction=variance_reduction,
                  rand_type=rand_type,
                  delayed_updates=delayed_update,
                  seed=seed)\
        .set_model(model_spars)\
        .set_prox(prox)
    solvers_sparse.append(solver)

    solver = SVRG(step=step, tol=tol, max_iter=max_iter, verbose=False,
                  print_every=10,
                  variance_reduction=variance_reduction,
                  rand_type=rand_type,
                  delayed_updates=delayed_update,
                  seed=seed)\
        .set_model(model_dense)\
        .set_prox(prox)
    solvers_dense.append(solver)

    title = "VR='" + variance_reduction + "'; RT='" + rand_type + "; UP='" \
            + delayed_update + "' " + "; prox=" + prox.name
    titles.append(title)


for solver_sparse, solver_dense, title in zip(solvers_sparse, solvers_dense,
                                              titles):
    print("-" * 80)
    print(title)
    solver_sparse.solve()
    solver_dense.solve()
    err = np.abs(solver_sparse.solution - solver_dense.solution).max()
    if err < 1e-10:
        print(err,
              np.linalg.norm(solver_sparse.solution[:-1]),
              np.linalg.norm(solver_dense.solution[:-1]))
    else:
        print(err,
              np.linalg.norm(solver_sparse.solution[:-1]),
              np.linalg.norm(solver_dense.solution[:-1]), '*****')


# for solver in solvers:
#     print(solver.solution)

#
# # In[24]:
#
# from tick.plot import plot_history
#
# _ = plot_history(solvers, rendering='matplotlib', log_scale=True,
#                  dist_min=True, x='n_iter', labels=titles)
# _ = plot_history(solvers, rendering='matplotlib', log_scale=True,
#                  dist_min=True, x='time', labels=titles)
#
#
# # In[25]:
#
# for solver, title in zip(solvers, titles):
#     plt.figure()
#     plt.stem(solver.solution)
#     plt.title(title)
#
#
# # In[26]:
#
# from sklearn.linear_model import LogisticRegression
#
#
# t1 = time()
# clf = LogisticRegression(solver='liblinear', tol=1e-7)
# clf.fit(A_spars, b)
# t2 = time()
#
# print(t2 - t1)
#
#
# # In[ ]:
#
#
#
#
# # In[32]:
#
# from tick.plot import stems
#
# stems([x_truth, x_spars, clf.coef_.ravel()])
#
#
# # In[17]:
#
# stems([x_dense - x_spars])
#
#
# # In[14]:
#
#
#
#
# # In[15]:
#
# from time import time
# t1 = time()
# clf.fit(A_spars, b)
# t2 = time()
# print(t2 - t1)
#
#
# # In[ ]:
#
#
#
