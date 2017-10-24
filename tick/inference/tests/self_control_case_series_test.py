import numpy as np
from tick.inference.tests.inference import InferenceTest
from tick.simulation import SimuSCCS
from tick.inference import LearnerSCCS


class Test(InferenceTest):
    def setUp(self):
        self.n_lags = 1
        self.seed = 42
        self.coeffs = np.log(np.array([2.1, 2.5,
                                       .8, .5]))
        n_features = int(len(self.coeffs) / (self.n_lags + 1))
        self.n_correlations = min(2, n_features - 1)
        # Create data
        sim = SimuSCCS(n_cases=500, n_intervals=10, n_features=n_features,
                       n_lags=self.n_lags, verbose=False, seed=self.seed,
                       coeffs=self.coeffs, n_correlations=self.n_correlations)
        _, self.features, self.labels, self.censoring, self.coeffs =\
            sim.simulate()

    def test_LearnerSCCS_coefficient_groups(self):
        coeffs = np.ones((4, 5))
        coeffs[0, 1:3] = 2
        coeffs[1, 2:] = 4
        coeffs[2, 0:2] = 0
        coeffs[3] = np.array([1, 2, 3, 4, 4])
        expected_equality_groups = [(1, 3), (3, 5),
                                    (5, 7), (7, 10),
                                    (10, 12), (12, 15),
                                    (18, 20)
                                    ]
        lrn = LearnerSCCS(n_lags=4, verbose=False)
        lrn._set("n_features", 4)
        equality_groups = lrn._detect_support(coeffs)
        self.assertEqual(expected_equality_groups, equality_groups)

    def test_LearnerSCCS_preprocess(self):
        # Just check that the preprocessing is running quickly
        lrn = LearnerSCCS(n_lags=self.n_lags, verbose=False,
                          penalty="None", feature_products=True)
        features, y, c = lrn._preprocess_data(self.features, self.labels,
                                                 self.censoring)
        # TODO: Check on small dummy data that preprocessing is working
        # TODO: fix feature products test
        pass

    def test_LearnerSCCS_fit(self):
        # TODO: a case with infinite features
        seed = 42
        n_lags = 2
        sim = SimuSCCS(n_cases=800, n_intervals=10, n_features=2,
                       n_lags=n_lags, verbose=False, seed=seed,
                       exposure_type='multiple_exposures')
        features, _, labels, censoring, coeffs = sim.simulate()
        lrn = LearnerSCCS(n_lags=n_lags, penalty="None", tol=0,
                          max_iter=10, verbose=False,
                          random_state=seed, feature_type='short')
        estimated_coeffs, _ = lrn.fit(features, labels, censoring)
        np.testing.assert_almost_equal(estimated_coeffs, coeffs, decimal=1)

    def test_LearnerSCCS_bootstrap_CI(self):
        lrn = LearnerSCCS(n_lags=self.n_lags, verbose=False)
        coeffs, _ = lrn.fit(self.features, self.labels, self.censoring)
        p_features, p_labels, p_censoring = lrn._preprocess_data(self.features,
                                                            self.labels,
                                                            self.censoring)
        bootstrap_ci = lrn._bootstrap(p_features, p_labels, p_censoring,
                                      5, .90)
        self.assertTrue(np.all(bootstrap_ci.lower_bound <= coeffs),
                        "lower bound of the confidence interval\
                               should be <= coeffs")
        self.assertTrue(np.all(coeffs <= bootstrap_ci.upper_bound),
                        "upper bound of the confidence interval\
                               should be >= coeffs")

    def test_LearnerSCCS_score(self):
        lrn = LearnerSCCS(n_lags=self.n_lags, verbose=False,
                          random_state=self.seed)
        lrn.fit(self.features, self.labels, self.censoring)
        self.assertEqual(lrn.score(),
                         lrn.score(self.features, self.labels, self.censoring))

    def test_LearnerSCCS_fit_KFold_CV(self):
        lrn = LearnerSCCS(n_lags=self.n_lags, verbose=False,
                          random_state=self.seed, penalty="TV-GroupL1",
                          strength_tv=1e-1, strength_group_l1=1e-1,
                          time_drift=False)
        lrn.fit(self.features, self.labels, self.censoring)
        score = lrn.score()
        tv_range = (-5, -1)
        groupl1_range = (-5, -1)
        # TODO: put this into the fit_kfold_cv method
        lrn.fit_kfold_cv(self.features, self.labels, self.censoring,
                         strength_tv_range=tv_range,
                         strength_group_l1_range=groupl1_range, n_cv_iter=4,
                         soft_cv=False)

        self.assertTrue(lrn.score() <= score)

    def test_LearnerSCCS_settings(self):
        # TODO: test all the settings here
        pass