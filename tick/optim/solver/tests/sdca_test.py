# License: BSD 3 clause

import unittest
import numpy as np

from tick.optim.model import ModelLogReg, ModelPoisReg
from tick.optim.prox import ProxL1, ProxElasticNet, ProxZero, ProxL2Sq
from tick.optim.solver import SDCA, SVRG
from tick.optim.solver.tests.solver import TestSolver
from tick.simulation import SimuPoisReg


class Test(TestSolver):
    def test_solver_sdca(self):
        """...Check SDCA solver for a Logistic regression with Ridge
        penalization and L1 penalization
        """
        solver = SDCA(l_l2sq=1e-5, max_iter=100, verbose=False, tol=0)
        self.check_solver(solver, fit_intercept=False, model="logreg",
                          decimal=1)

    def compare_solver_sdca(self):
        # Now a specific test with a real prox for SDCA
        np.random.seed(12)
        n_samples = Test.n_samples
        n_features = Test.n_features

        for fit_intercept in [False]:
            y, X, coeffs0, interc0 = TestSolver.generate_logistic_data(
                n_features, n_samples)

            model = ModelLogReg(fit_intercept=fit_intercept).fit(X, y)
            ratio = 0.5
            l_enet = 1e-2

            # SDCA "elastic-net" formulation is different from elastic-net
            # implementation
            l_l2_sdca = ratio * l_enet
            l_l1_sdca = (1 - ratio) * l_enet
            sdca = SDCA(l_l2sq=l_l2_sdca, max_iter=100, verbose=False, tol=0,
                        seed=Test.sto_seed)
            sdca.set_model(model)
            prox_l1 = ProxL1(l_l1_sdca)
            sdca.set_prox(prox_l1)
            coeffs_sdca = sdca.solve()

            # Compare with SVRG
            svrg = SVRG(max_iter=100, verbose=False, tol=0,
                        seed=Test.sto_seed)
            svrg.set_model(model)
            prox_enet = ProxElasticNet(l_enet, ratio)
            svrg.set_prox(prox_enet)
            coeffs_svrg = svrg.solve(step=0.1)

            np.testing.assert_allclose(coeffs_sdca, coeffs_svrg)

    def test_sdca_sparse_and_dense_consistency(self):
        """...Test SDCA can run all glm models and is consistent with sparsity
        """

        def create_solver():
            return SDCA(max_iter=1, verbose=False, l_l2sq=1e-3,
                        seed=TestSolver.sto_seed)

        self._test_solver_sparse_and_dense_consistency(create_solver)

    def test_sdca_identity_poisreg(self):
        """...Test SDCA on specific case of Poisson regression with
        indentity link
        """
        np.set_printoptions(precision=4, linewidth=200)

        l_l2sq = 1e-1
        n_samples = 190
        n_features = 10

        np.random.seed(123)
        weight0 = np.random.rand(n_features)
        features = np.random.rand(n_samples, n_features)

        simu = SimuPoisReg(weight0, intercept=None,
                           features=features, n_samples=n_samples,
                           link='identity')
        features, labels = simu.simulate()
        labels[labels != 0] = 10

        model = ModelPoisReg(fit_intercept=False, link='identity')
        model.fit(features, labels)

        sdca = SDCA(l_l2sq=l_l2sq, max_iter=100, verbose=True, tol=1e-14,
                    seed=Test.sto_seed, print_every=10).set_model(model).set_prox(ProxZero())

        sdca.history.set_minimizer(weight0)

        primal, dual = model.init_sdca_primal_dual_variables(l_l2sq, init_dual=None)
        print(primal)
        sdca._solver.init_stored_variables(primal.copy(), dual.copy())
        sdca.solve()

        print('original coeffs')
        print(weight0)

        print("solver primal")
        print(sdca._solver.get_primal_vector())



        svrg = SVRG(max_iter=100, verbose=True, tol=1-10,
                    seed=Test.sto_seed)
        svrg.set_model(model)
        prox_l2sq = ProxL2Sq(l_l2sq)
        svrg.set_prox(prox_l2sq)
        coeffs_svrg = svrg.solve(0.5 * np.ones(model.n_coeffs), step=1e-1)

        print("solver sqvrg")
        print(coeffs_svrg)

        print('P(w) = ', svrg.objective(coeffs_svrg))
        print('P(w) = ', sdca.objective(coeffs_svrg))
        print('P(w) = ', sdca.objective(sdca._solver.get_primal_vector()))
        print('D(a) = ', sdca.dual_objective(sdca._solver.get_dual_vector()))

        from scipy.optimize import approx_fprime
        print(approx_fprime(sdca._solver.get_primal_vector(), sdca.objective, 1e-10))
        print(approx_fprime(sdca._solver.get_dual_vector(), sdca.dual_objective,
                            1e-10).max())


if __name__ == '__main__':
    unittest.main()
