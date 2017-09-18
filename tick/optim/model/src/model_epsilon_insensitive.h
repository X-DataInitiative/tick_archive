#ifndef TICK_OPTIM_MODEL_SRC_EPSILONINSENSITIVE_H_
#define TICK_OPTIM_MODEL_SRC_EPSILONINSENSITIVE_H_

// License: BSD 3 clause

#include "model_generalized_linear.h"
#include "model_lipschitz.h"

#include <cereal/types/base_class.hpp>

class ModelEpsilonInsensitive : public virtual ModelGeneralizedLinear {
 private:
  double threshold;

 public:
  ModelEpsilonInsensitive(const SBaseArrayDouble2dPtr features,
                          const SArrayDoublePtr labels,
                          const bool fit_intercept,
                          const double threshold,
                          const int n_threads = 1);

  const char *get_class_name() const override;

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  virtual double get_threshold(void) const {
    return threshold;
  }

  virtual void set_threshold(const double threshold) {
    if (threshold <= 0.) {
      TICK_ERROR("threshold must be > 0");
    } else {
      this->threshold = threshold;
    }
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
    ar(cereal::make_nvp("ModelLipschitz", cereal::base_class<ModelLipschitz>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelEpsilonInsensitive, cereal::specialization::member_serialize)

#endif  // TICK_OPTIM_MODEL_SRC_EPSILONINSENSITIVE_H_
