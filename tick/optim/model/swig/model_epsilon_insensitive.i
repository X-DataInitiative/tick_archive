// License: BSD 3 clause

%{
#include "model_epsilon_insensitive.h"
%}


class ModelEpsilonInsensitive : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelEpsilonInsensitive(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);
};
