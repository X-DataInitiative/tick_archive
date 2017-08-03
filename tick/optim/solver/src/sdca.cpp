// License: BSD 3 clause

//
// Created by Martin Bompaire on 22/10/15.
//

#include <prox_l2sq.h>
#include "sdca.h"

SDCA::SDCA(double l_l2sq,
           ulong epoch_size,
           double tol,
           RandType rand_type,
           int seed
) : StoSolver(epoch_size, tol, rand_type, seed), l_l2sq(l_l2sq) {
  stored_variables_ready = false;
}

void SDCA::set_model(ModelPtr model) {
  StoSolver::set_model(model);
  this->model = model;
  stored_variables_ready = false;
}

void SDCA::reset() {
  StoSolver::reset();
  init_stored_variables();
}

void SDCA::init_stored_variables() {
  // TODO(martin) check not linear Poisreg nor Hawkes, otherwise raise error

  n_coeffs = model->get_n_coeffs();

  if (dual_vector.size() != rand_max)
    dual_vector = ArrayDouble(rand_max);

  if (tmp_primal_vector.size() != n_coeffs)
    tmp_primal_vector = ArrayDouble(n_coeffs);

  dual_vector.init_to_zero();
  tmp_primal_vector.init_to_zero();

  init_stored_variables(tmp_primal_vector, dual_vector);
}

void SDCA::init_stored_variables(ArrayDouble &original_primal_vector,
                                 ArrayDouble &original_dual_vector) {
  n_coeffs = model->get_n_coeffs();

  if (original_primal_vector.size() != n_coeffs)
    throw std::runtime_error("Primal vector should have a size equal to the number of coefficients"
                               " of the model");
  if (original_dual_vector.size() != rand_max)
    throw std::runtime_error("Dual vector should have a size equal to the number of samples"
                               " of the model");

  if (delta.size() != rand_max)
    delta = ArrayDouble(rand_max);

  // Don't copy if it is the same array, this might happen in init_stored_variables()
//  if (&dual_vector != &original_dual_vector)
    dual_vector = original_dual_vector;

//  if (&tmp_primal_vector != &original_dual_vector)
    tmp_primal_vector = original_primal_vector;
  iterate = original_primal_vector;

  std::cout << "Initialized primal to" << std::endl;
  tmp_primal_vector.print();
  iterate.print();

  delta.init_to_zero();
  stored_variables_ready = true;
}

void SDCA::solve() {
  if (!stored_variables_ready) {
    init_stored_variables();
  }

  ulong i;
  double _1_over_lbda_n = 1 / (l_l2sq * rand_max);
  ulong start_t = t;

  std::cout << "_1_over_lbda_n = " << _1_over_lbda_n << std::endl;

  for (t = start_t; t < start_t + epoch_size; ++t) {
    // Pick i uniformly at random
    i = get_next_i();

    // Maximize the dual coordinate i
    double delta_dual_i = model->sdca_dual_min_i(i, dual_vector, iterate, delta, l_l2sq);

    // Update the dual variable
    dual_vector[i] += delta_dual_i;

    // Keep the last ascent seen for warm-starting sdca_dual_min_i
    delta[i] = delta_dual_i;

    // Update the primal variable
    BaseArrayDouble features_i = model->get_sdca_features(i);
//    std::cout << "did get "<< i << std::endl;

    if (model->use_intercept()) {
      ArrayDouble primal_features = view(tmp_primal_vector, 0, features_i.size());
      primal_features.mult_incr(features_i, delta_dual_i * _1_over_lbda_n);
      tmp_primal_vector[model->get_n_features()] += delta_dual_i * _1_over_lbda_n;
    } else {
      std::cout << "features_i" << std::endl;
      features_i.print();
      tmp_primal_vector.mult_incr(features_i, delta_dual_i * _1_over_lbda_n);
      std::cout << "primal" << std::endl;
      tmp_primal_vector.print();
    }

    // Call prox on the primal variable
    prox->call(tmp_primal_vector, 1. / l_l2sq, iterate);
    std::cout << "primal after prox" << std::endl;
    iterate.print();
    std::cout << "new dual" << std::endl;
    dual_vector.print();
    std::cout << "-------" << std::endl;
  }
}
