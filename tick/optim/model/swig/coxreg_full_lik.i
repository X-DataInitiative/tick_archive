// License: BSD 3 clause

%{
#include "coxreg_full_lik.h"
%}

enum class BaselineType {
    exponential = 0,
    weibull,
    histogram
};

class ModelCoxRegFullLik : public Model {
 public:

  ModelCoxRegFullLik(const SBaseArrayDouble2dPtr features,
                     const SArrayDoublePtr times,
                     const SArrayUShortPtr censoring,
                     const BaselineType baseline,
                     const int n_threads);

  void set_baseline(BaselineType baseline);

  BaselineType get_baseline();
};
