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
  n_samples = features->n_rows();
  n_features = features->n_cols();

}

void ModelCoxRegFullLik::compute_grad_i(const ulong i, const ArrayDouble &coeffs,
                                        ArrayDouble &out, const bool fill) {
  const BaseArrayDouble x_i = get_features(i);

  if (baseline == BaselineType::exponential) {
    ArrayDouble coeffs_without_baseline = view(coeffs, 0, n_features);
    double inner_prod = get_inner_prod(i, coeffs_without_baseline);
    double baseline = coeffs[n_features];
    // std::cout << "coeffs_without_baseline" << std::endl;
    // coeffs_without_baseline.print();
    // std::cout << "baseline" << std::endl;
    // std::cout << baseline << std::endl;
    if (baseline < 0) {
      TICK_ERROR("In ModelCoxRegFullLik::loss_i with a negative baseline")
    }

    // The number in front of the features that leads to the gradient with respect to
    double alpha_i = baseline * get_time(i) * exp(inner_prod);
    double grad_baseline = get_time(i) * exp(inner_prod);
    if (get_censoring(i) == 1) {
      alpha_i += inner_prod;
      grad_baseline += 1 / baseline;
    }
    ArrayDouble out_without_baseline = view(out, 0, n_features);
    if (fill) {
      out_without_baseline.mult_fill(x_i, alpha_i);
      out[n_features] = grad_baseline;
    } else {
      out_without_baseline.mult_incr(x_i, alpha_i);
      out[n_features] += grad_baseline;
    }
  }
}

double ModelCoxRegFullLik::loss_i(const ulong i,
                                  const ArrayDouble &coeffs) {
  if (baseline == BaselineType::exponential) {
    ArrayDouble coeffs_without_baseline = view(coeffs, 0, n_features);
    double inner_prod = get_inner_prod(i, coeffs_without_baseline);
    double baseline = coeffs[n_features];
    // std::cout << "coeffs_without_baseline" << std::endl;
    // coeffs_without_baseline.print();
    // std::cout << "baseline" << std::endl;
    // std::cout << baseline << std::endl;
    if (baseline < 0) {
      TICK_ERROR("In ModelCoxRegFullLik::loss_i with a negative baseline")
    }
    double loss = 0;
    loss += baseline * get_time(i) * exp(inner_prod);
    if (get_censoring(i) == 1) {
      loss += log(baseline) + inner_prod;
    }
    return loss;
  } else {
    if (baseline == BaselineType::weibull) {

    } else {
      // Histogram case
    }
  }
}