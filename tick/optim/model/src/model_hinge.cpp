// License: BSD 3 clause

#include "model_hinge.h"

ModelHinge::ModelHinge(const SBaseArrayDouble2dPtr features,
                       const SArrayDoublePtr labels,
                       const bool fit_intercept,
                       const double threshold,
                       const int n_threads)

    : ModelGeneralizedLinear(features,
                             labels,
                             fit_intercept,
                             n_threads),
      ModelLipschitz() {
  this->threshold = threshold;
}

const char *ModelHinge::get_class_name() const {
  return "ModelHinge";
}

double ModelHinge::loss_i(const ulong i,
                          const ArrayDouble &coeffs) {
  const double z = get_label(i) * get_inner_prod(i, coeffs);

  if (z <= threshold) {
    return threshold - z;
  } else {
    return 0.;
  }
}

double ModelHinge::grad_i_factor(const ulong i,
                                 const ArrayDouble &coeffs) {
  const double y = get_label(i);
  if (y * get_inner_prod(i, coeffs) <= threshold) {
    return y;
  } else {
    return 0;
  }
}

void ModelHinge::compute_lip_consts() {
  if (ready_lip_consts) {
    return;
  } else {
    compute_features_norm_sq();
    lip_consts = ArrayDouble(n_samples);
    for (ulong i = 0; i < n_samples; ++i) {
      if (fit_intercept) {
        lip_consts[i] = features_norm_sq[i] + 1;
      } else {
        lip_consts[i] = features_norm_sq[i];
      }
    }
  }
}
