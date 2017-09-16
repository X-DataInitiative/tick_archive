// License: BSD 3 clause

%{
#include "model_hinge.h"
%}


class ModelHinge : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelHinge(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const double threshold,
              const int n_threads);
};
