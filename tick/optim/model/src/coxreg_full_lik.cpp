// License: BSD 3 clause

#include "coxreg_full_lik.h"

ModelCoxRegFullLik::ModelCoxRegFullLik(const SBaseArrayDouble2dPtr features,
                                       const SArrayDoublePtr times,
                                       const SArrayUShortPtr censoring,
                                       const BaselineType baseline,
                                       const int n_threads)

    : ModelGeneralizedLinear(features, times, false, n_threads),
      baseline(baseline), censoring(censoring) {
  n_bins = 0;

}

void ModelCoxRegFullLik::grad_i(const ulong i,
                                const ArrayDouble &coeffs,
                                ArrayDouble &out) {

}

void ModelCoxRegFullLik::inc_grad_i(const ulong i,
                                    ArrayDouble &out,
                                    const ArrayDouble &coeffs) {

}

void ModelCoxRegFullLik::grad(const ArrayDouble &coeffs,
                              ArrayDouble &out) {

}

double ModelCoxRegFullLik::loss(const ArrayDouble &coeffs) {
  return 0;
}

