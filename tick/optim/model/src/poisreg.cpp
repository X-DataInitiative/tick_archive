// License: BSD 3 clause

//
// Created by Martin Bompaire on 21/10/15.
//

#include "poisreg.h"

ModelPoisReg::ModelPoisReg(const SBaseArrayDouble2dPtr features,
                           const SArrayDoublePtr labels,
                           const LinkType link_type,
                           const bool fit_intercept,
                           const int n_threads)
  : ModelGeneralizedLinear(features,
                           labels,
                           fit_intercept,
                           n_threads),
    link_type(link_type), non_zero_label_map_computed(false) {}

double ModelPoisReg::loss_i(const ulong i, const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  switch (link_type) {
    case LinkType::exponential: {
      double y_i = get_label(i);
      return exp(z) - y_i * z + std::lgamma(y_i + 1);
    }
    case LinkType::identity: {
      double y_i = get_label(i);
      return z - y_i * log(z) + std::lgamma(y_i + 1);
    }
    default:throw std::runtime_error("Undefined link type");
  }
}

double ModelPoisReg::grad_i_factor(const ulong i, const ArrayDouble &coeffs) {
  const double z = get_inner_prod(i, coeffs);
  switch (link_type) {
    case LinkType::exponential: {
      return exp(z) - get_label(i);
    }
    case LinkType::identity: {
      return 1 - get_label(i) / z;
    }
    default:throw std::runtime_error("Undefined link type");
  }
}

double ModelPoisReg::sdca_dual_min_i(const ulong i,
                                     const ArrayDouble &dual_vector,
                                     const ArrayDouble &primal_vector,
                                     const ArrayDouble &previous_delta_dual,
                                     const double l_l2sq) {
  if (link_type == LinkType::identity) {
    return sdca_dual_min_i_identity(i, dual_vector, primal_vector, previous_delta_dual, l_l2sq);
  } else {
    return sdca_dual_min_i_exponential(i, dual_vector, primal_vector, previous_delta_dual, l_l2sq);
  }

}

double ModelPoisReg::sdca_dual_min_i_exponential(const ulong i,
                                                 const ArrayDouble &dual_vector,
                                                 const ArrayDouble &primal_vector,
                                                 const ArrayDouble &previous_delta_dual,
                                                 double l_l2sq) {
  compute_features_norm_sq();
  double epsilon = 1e-1;

  double normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_samples);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_samples);
  }
  const double primal_dot_features = get_inner_prod(i, primal_vector);
  double delta_dual = previous_delta_dual[i];
  const double dual = dual_vector[i];
  const double label = get_label(i);
  double new_dual{0.};

  for (int j = 0; j < 10; ++j) {
    new_dual = dual + delta_dual;

    // Check we are in the correct bounds
    if (new_dual >= label) {
      new_dual = label - epsilon;
      delta_dual = new_dual - dual;
      epsilon *= 1e-1;
    }

    // Do newton descent
    // Poisson loss part
    double f_prime = -log(label - new_dual);
    double f_second = 1. / (label - new_dual);

    // Ridge regression part
    f_prime += normalized_features_norm * delta_dual + primal_dot_features;
    f_second += normalized_features_norm;

    delta_dual -= f_prime / f_second;
    new_dual = dual + delta_dual;

    if (std::abs(f_prime / f_second) < 1e-10) {
      break;
    }
  }
  // Check we are in the correct bounds
  if (new_dual >= label) {
    new_dual = label - epsilon;
    delta_dual = new_dual - dual;
  }

  return delta_dual;
}

ulong ModelPoisReg::get_non_zero_i(ulong i) {
  if (!non_zero_label_map_computed) init_non_zero_label_map();
  return non_zero_label_map[i];
}

void ModelPoisReg::init_non_zero_label_map() {
  non_zero_label_map.clear();
  ulong n_non_zeros = 0;
  for (ulong i = 0; i < get_n_samples(); ++i) {
    if (get_label(i) != 0) {
      non_zero_label_map[n_non_zeros] = i;
      n_non_zeros++;
    }
  }
  n_non_zeros_labels = n_non_zeros;
}

BaseArrayDouble ModelPoisReg::get_sdca_features(const ulong i) {
  if (!non_zero_label_map_computed) init_non_zero_label_map();
  get_features(non_zero_label_map[i]);
  return get_features(non_zero_label_map[i]);
}

double ModelPoisReg::sdca_dual_min_i_identity(const ulong rand_i,
                                              const ArrayDouble &dual_vector,
                                              const ArrayDouble &primal_vector,
                                              const ArrayDouble &previous_delta_dual,
                                              const double l_l2sq) {

  if (fit_intercept) {
    TICK_ERROR("SDCA not implemented with fit intercept yet");
  }

  if (!ready_features_norm_sq) {
    compute_features_norm_sq();
  }

  ulong i = get_non_zero_i(rand_i);

  const double label = get_label(i);
  const double dual = dual_vector[rand_i];
  if (label == 0) {
    TICK_ERROR("Labels 0 should not be considered in SDCA");
  }

  double normalized_features_norm = features_norm_sq[i] / (l_l2sq * n_non_zeros_labels);
  if (use_intercept()) {
    normalized_features_norm += 1. / (l_l2sq * n_non_zeros_labels);
  }
  double primal_dot_features = get_inner_prod(i, primal_vector);

  double tmp = dual * normalized_features_norm - primal_dot_features;
  double new_dual = (std::sqrt(tmp * tmp + 4 * label * normalized_features_norm) + tmp) / (2
    * normalized_features_norm);

  return new_dual - dual;

}
