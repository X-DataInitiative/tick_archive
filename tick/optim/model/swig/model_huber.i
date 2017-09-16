// License: BSD 3 clause

%{
#include "model_huber.h"
%}


class ModelHuber : public virtual ModelGeneralizedLinear,
                    public ModelLipschitz {
 public:

  ModelHuber(const SBaseArrayDouble2dPtr features,
              const SArrayDoublePtr labels,
              const bool fit_intercept,
              const int n_threads);
};
