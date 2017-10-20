// License: BSD 3 clause

#include "coxreg_full_lik.h"



ModelCoxRegFullLik::ModelCoxRegFullLik(const SBaseArrayDouble2dPtr features,
                   const SArrayDoublePtr times,
                   const SArrayUShortPtr censoring,
                   const BaselineType baseline,
                                       const int n_threads)

    : ModelGeneralizedLinear(features, labels, false, n_threads),
      baseline(baseline) {

}


double ModelCoxRegFullLik::grad_i_factor(const ulong i,
                                         const ArrayDouble &coeffs) {

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

}


double ModelCoxRegFullLik::get_inner_prod(const ulong i,
                                          const ArrayDouble &coeffs) {

}