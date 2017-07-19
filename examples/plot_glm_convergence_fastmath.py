"""
============================================
Generalized linear models solver convergence
============================================

This example illustrates the opimization of three linear models:
    * Linear regression (`tick.optim.model.ModelLinReg`)
    * Logistic regression (`tick.optim.model.ModelLogReg`)
    * Poisson regression (`tick.optim.model.ModelPoisReg`)

with five different solvers:
    * LBFGS (`tick.optim.solver.BFGS`)
    * SVRG (`tick.optim.solver.SVRG`)
    * SDCA (`tick.optim.solver.SDCA`)
    * GD (`tick.optim.solver.GD`)
    * AGD (`tick.optim.solver.AGD`)
"""

import matplotlib.pyplot as plt
from tick.plot import plot_history
import numpy as np
import scipy
from itertools import product
from tick.optim.model import ModelLinReg, ModelLogReg, ModelPoisReg
from tick.optim.solver import SDCA, SVRG, BFGS, GD, AGD
from tick.optim.prox import ProxZero, ProxL2Sq
from tick.simulation import SimuLinReg, SimuLogReg, SimuPoisReg

seed = 1398
np.random.seed(seed)


def create_model(n_samples, n_features, sparse, nnz=None, with_intercept=True):
    if sparse:
        weights = np.random.randn(nnz)
    else:
        weights = np.random.randn(n_features)

    intercept = None
    if with_intercept:
        intercept = np.random.normal()

    simulator = SimuLogReg(weights, intercept=intercept,
                           n_samples=n_samples, verbose=False)
    nnz_features, labels = simulator.simulate()

    if sparse:
        all_cols = np.arange(n_features, dtype=int)
        col_indices = np.random.choice(all_cols, size=nnz * n_samples)

        row_indices = np.tile(np.arange(n_samples), (nnz, 1)).T
        row_indices = row_indices.ravel()

        nnz_features = nnz_features.ravel()

        features = scipy.sparse.csr_matrix((nnz_features,
                                           (row_indices, col_indices)))
    else:
        features = nnz_features

    model = ModelLogReg(fit_intercept=with_intercept)

    model.fit(features, labels)
    return model


def run_solvers(model, l_l2sq):
    try:
        svrg_step = 1. / model.get_lip_max()
    except AttributeError:
        svrg_step = 1e-3

    bfgs = BFGS(verbose=True, tol=1e-13)
    bfgs.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    bfgs.solve()
    bfgs.history.set_minimizer(bfgs.solution)
    bfgs.history.set_minimum(bfgs.objective(bfgs.solution))
    bfgs.solve()

    svrg = SVRG(step=svrg_step, verbose=True, tol=1e-10, seed=seed)
    svrg.set_model(model).set_prox(ProxL2Sq(l_l2sq))
    svrg.history.set_minimizer(bfgs.solution)
    svrg.history.set_minimum(bfgs.objective(bfgs.solution))
    svrg.solve()

    sdca = SDCA(l_l2sq, verbose=True, seed=seed, tol=1e-10)
    sdca.set_model(model).set_prox(ProxZero())
    sdca.history.set_minimizer(bfgs.solution)
    sdca.history.set_minimum(bfgs.objective(bfgs.solution))
    sdca.solve()

    return bfgs, svrg, sdca

n_samples = 10000
l_l2sq = 1 / np.sqrt(n_samples)

for sparse in [True, False]:
    if sparse:
        n_features = 100000
        nnz = int(0.01 * n_features)
    else:
        n_features = 1000
        nnz = None

    model = create_model(n_samples, n_features, sparse, nnz=nnz)

    bfgs, svrg, sdca = run_solvers(model, l_l2sq)
    fig = plot_history([bfgs, svrg, sdca],
                       dist_min=True, log_scale=True, x='time', show=False)

    fig_title = 'with fast math'
    if sparse:
        fig_title += ' (sparse)'
    else:
        fig_title += ' (dense)'

    plt.title(fig_title)
    plt.savefig(fig_title, dpi=200)
