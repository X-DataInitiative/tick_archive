# License: BSD 3 clause

import tick.base

from .online_forest_classifier import OnlineForestClassifier
from .online_forest_regressor import OnlineForestRegressor

__all__ = [
    "OnlineForestClassifier",
    "OnlineForestRegressor"
]
