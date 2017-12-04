# License: BSD 3 clause

from tick.linear_model.linreg import ModelLinReg
from .absolute_regression import ModelAbsoluteRegression
from .coxreg_partial_lik import ModelCoxRegPartialLik
from .epsilon_insensitive import ModelEpsilonInsensitive
from .hawkes_fixed_expkern_leastsq import ModelHawkesFixedExpKernLeastSq
from .hawkes_fixed_expkern_loglik import ModelHawkesFixedExpKernLogLik
from .hawkes_fixed_sumexpkern_leastsq import ModelHawkesFixedSumExpKernLeastSq
from .hawkes_fixed_sumexpkern_loglik import ModelHawkesFixedSumExpKernLogLik
from .huber import ModelHuber
from .linreg_with_intercepts import ModelLinRegWithIntercepts
from .modified_huber import ModelModifiedHuber
from .sccs import ModelSCCS

__all__ = ["ModelLinReg",
           "ModelLinRegWithIntercepts",
           "ModelLogReg",
           "ModelPoisReg",
           'ModelHinge',
           'ModelSmoothedHinge',
           'ModelQuadraticHinge',
           'ModelHuber',
           'ModelModifiedHuber',
           'ModelEpsilonInsensitive',
           'ModelAbsoluteRegression',
           "ModelCoxRegPartialLik",
           "ModelHawkesFixedExpKernLogLik",
           "ModelHawkesFixedSumExpKernLogLik",
           "ModelHawkesFixedExpKernLeastSq",
           "ModelHawkesFixedSumExpKernLeastSq",
           "ModelSCCS"
           ]
