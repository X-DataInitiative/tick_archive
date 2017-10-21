# License: BSD 3 clause

import numpy as np

from tick.optim.model.base import Model, ModelFirstOrder, ModelGeneralizedLinear
from tick.preprocessing.utils import safe_array
from tick.optim.model.build.model import ModelCoxRegFullLik \
    as _ModelCoxRegFullLik
from .build.model import BaselineType_exponential as exponential
from .build.model import BaselineType_histogram as histogram
from .build.model import BaselineType_weibull as weibull


class ModelCoxRegFullLik(ModelFirstOrder, ModelGeneralizedLinear):
    """Full likelihood of the Cox regression model (proportional
    hazards).
    This class gives first order information (gradient and loss) for
    this model.

    Parameters
    ----------
    baseline : {'exponential', 'weibull', 'histogram'}, default='exponential'
        The model used for the baseline intensity of the model.

    n_bins : `int`
        Number of bins to use when baseline='histogram'

    Attributes
    ----------
    features : `numpy.ndarray`, shape=(n_samples, n_features), (read-only)
        The features matrix

    times : `numpy.ndarray`, shape = (n_samples,), (read-only)
        Obverved times

    censoring : `numpy.ndarray`, shape = (n_samples,), (read-only)
        Boolean indicator of censoring of each sample.
        ``True`` means true failure, namely non-censored time

    n_samples : `int` (read-only)
        Number of samples

    n_features : `int` (read-only)
        Number of features

    n_failures : `int` (read-only)
        Number of true failure times

    n_coeffs : `int` (read-only)
        Total number of coefficients of the model

    censoring_rate : `float`
        The censoring_rate (percentage of ???)

    Notes
    -----
    There is no intercept in this model
    """

    _attrinfos = {
        "features": {
            "writable": False
        },
        "times": {
            "writable": False
        },
        "censoring": {
            "writable": False
        },
        "n_samples": {
            "writable": False
        },
        "n_features": {
            "writable": False
        },
        "n_failures": {
            "writable": False
        },
        "censoring_rate": {
            "writable": False
        },
        "_baseline": {
            "writable": False,
            "cpp_setter": "set_baseline"
        }
    }

    # TODO: times must be aligned with zero
    def __init__(self, baseline: str = 'exponential', n_threads=1):
        ModelFirstOrder.__init__(self)
        ModelGeneralizedLinear.__init__(self, False)
        self.features = None
        self.times = None
        self.censoring = None
        self.n_samples = None
        self.n_features = None
        self.n_failures = None
        self.censoring_rate = None
        self._baseline = None
        self._model = None
        self.baseline = baseline
        self.n_threads = n_threads
        self.n_bins = None

    def fit(self, features: np.ndarray, times: np.array, censoring: np.array):
        """Set the data into the model object

        Parameters
        ----------
        features : `numpy.ndarray`, shape=(n_samples, n_features)
            The features matrix

        times : `numpy.array`, shape = (n_samples,)
            Observed times

        censoring : `numpy.array`, shape = (n_samples,)
            Indicator of censoring of each sample.
            ``True`` means true failure, namely non-censored time.
            dtype must be unsigned short

        Returns
        -------
        output : `ModelCoxRegPartialLik`
            The current instance with given data
        """
        # The fit from Model calls the _set_data below
        return Model.fit(self, features, times, censoring)

    def _set_data(self, features: np.ndarray, times: np.array,
                  censoring: np.array):
        n_samples, n_features = features.shape
        if n_samples != times.shape[0]:
            raise ValueError(("Features has %i samples while times "
                              "have %i" % (n_samples, times.shape[0])))
        if n_samples != censoring.shape[0]:
            raise ValueError(("Features has %i samples while censoring "
                              "have %i" % (n_samples,
                                           censoring.shape[0])))

        features = safe_array(features)
        times = safe_array(times)
        censoring = safe_array(censoring, np.ushort)

        self._set("features", features)
        self._set("times", times)
        self._set("censoring", censoring)
        self._set("n_samples", n_samples)
        self._set("n_features", n_features)
        self._set("_model", _ModelCoxRegFullLik(self.features, self.times,
                                                self.censoring, self._baseline,
                                                self.n_threads))

    def _grad(self, coeffs: np.ndarray, out: np.ndarray) -> None:
        self._model.grad(coeffs, out)

    def _loss(self, coeffs: np.ndarray) -> float:
        return self._model.loss(coeffs)

    def _get_n_coeffs(self, *args, **kwargs):
        if self.baseline == 'exponential':
            return self.n_features + 1
        elif self.baseline == 'weibull':
            return self.n_features + 2
        else:
            return self.n_features + self.n_bins

    # TODO: self.baseline returns the name of the baseline
    # self.baseline = sets the name of the baseline and check that it's
    # indeed in the right value
    # If self._model is not none, then we set the model
    # We only set the baseline when creating the model
    # How to use the C++ settter in this case ?

    @property
    def baseline(self):
        _baseline = self._baseline
        if _baseline == exponential:
            return 'exponential'
        elif _baseline == weibull:
            return 'weibull'
        else:
            return 'histogram'

    @baseline.setter
    def baseline(self, value):
        if value == 'exponential':
            self._set('_baseline', exponential)
        elif value == 'weibull':
            self._set('_baseline', weibull)
        elif value == 'histogram':
            self._set('_baseline', histogram)
        else:
            raise ValueError("``baseline`` must be either 'exponential' or "
                             "'weibull' or 'histogram'.")

    @property
    def _epoch_size(self):
        return self.n_samples

    @property
    def _rand_max(self):
        # This allows to obtain the range of the random sampling when
        # using a stochastic optimization algorithm
        return self.n_samples

    def _as_dict(self):
        dd = ModelFirstOrder._as_dict(self)
        del dd["features"]
        del dd["times"]
        del dd["censoring"]
        return dd
