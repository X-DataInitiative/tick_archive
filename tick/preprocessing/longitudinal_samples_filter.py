import numpy as np
from operator import itemgetter
from tick.preprocessing.base import LongitudinalPreprocessor


class LongitudinalSamplesFilter(LongitudinalPreprocessor):
    _attrinfos = {
        "_mask": {"writable": False},
        "_n_active_patients": {"writable": False},
        "_n_patients": {"writable": False},
    }

    # TODO: DOCUMENTATION
    def __init__(self, n_jobs=-1):
        LongitudinalPreprocessor.__init__(self, n_jobs=n_jobs)
        self._mask = None
        self._n_active_patients = None
        self._n_patients = None

    def fit(self, features, labels, censoring):
        nnz = [len(np.nonzero(arr)[0]) > 0 for arr in labels]
        self._set('_mask', [idx for idx, feat in enumerate(features)
                            if feat.sum() > 0 and nnz[idx]])
        self._set('_n_active_patients', len(self._mask))
        self._set('_n_patients', len(features))

        return self

    def transform(self, features, labels, censoring):
        if self._n_active_patients == 0:
            raise ValueError("There should be at least one positive sample per\
                 batch with nonzero_features. Please check the input data.")
        if self._n_active_patients < self._n_patients:
            # TODO: raise warning?
            features_filter = itemgetter(*self._mask)
            features = features_filter(features)
            labels = features_filter(labels)
            censoring = censoring[self._mask]

        return features, labels, censoring

