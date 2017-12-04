# License: BSD 3 clause

import tick.base

from .linreg import ModelLinReg
from .logreg import ModelLogReg
from .poisreg import ModelPoisReg
from .hinge import ModelHinge
from .smoothed_hinge import ModelSmoothedHinge
from .quadratic_hinge import ModelQuadraticHinge

__all__ = [
    'ModelLinReg',
    'ModelLogReg',
    'ModelPoisReg',
    'ModelHinge',
    'ModelSmoothedHinge',
    'ModelQuadraticHinge'
]
