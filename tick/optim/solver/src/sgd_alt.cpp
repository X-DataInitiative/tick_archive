//
// Created by Martin Bompaire on 22/10/15.
//

#include "sgd_alt.h"

SGDAlt::SGDAlt(ulong epoch_size,
               double tol,
               RandType rand_type,
               double step,
               int seed)
    : StoSolver(epoch_size, tol, rand_type, seed),
      step(step) {}

void SGDAlt::solve() {
  if (model->is_sparse()) {
    solve_sparse();
  } else {
    // Dense case
    ArrayDouble grad_sum(iterate.size());
    ArrayDouble grad(iterate.size());
    grad.init_to_zero();

    const ulong batch_size = 4;

    ArrayULong rand_indices(epoch_size * 2);

    for (ulong i = 0; i < rand_indices.size(); ++i) {
      rand_indices[i] = get_next_i();
    }

    const ulong start_t = t;
    ulong iteration = 0;

    for (t = start_t; t < start_t + epoch_size; t += 0) {
      grad_sum.init_to_zero();
      for (ulong j = 0; j < batch_size; ++j) {
        model->grad_i(rand_indices[iteration++], iterate, grad);

        for (ulong f_i = 0; f_i < iterate.size(); ++f_i) {
          step_t = get_step_t();
          grad_sum[f_i] += grad[f_i] * (-step_t);
        }

        t += 1;
      }

      for (ulong f_i = 0; f_i < iterate.size(); ++f_i) {
        grad_sum[f_i] *= (1.0) / batch_size;
      }

      step_t = get_step_t();
      iterate.mult_incr(grad_sum, 1.0);

      prox->call(iterate, step_t, iterate);
    }
  }
}

void SGDAlt::solve_sparse() {
  // The model is sparse, so it is a ModelGeneralizedLinear and the iteration looks a
  // little bit different
  ulong n_features = model->get_n_features();
  bool use_intercept = model->use_intercept();

  ulong start_t = t;
  for (t = start_t; t < start_t + epoch_size; ++t) {
    ulong i = get_next_i();
    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);
    // Gradient factor
    double alpha_i = model->grad_i_factor(i, iterate);
    // Update the step
    double step_t = get_step_t();
    double delta = -step_t * alpha_i;
    if (use_intercept) {
      // Get the features vector, which is sparse here
      ArrayDouble iterate_no_interc = view(iterate, 0, n_features);
      iterate_no_interc.mult_incr(x_i, delta);
      iterate[n_features] += delta;
    } else {
      // Stochastic gradient descent step
      iterate.mult_incr(x_i, delta);
    }
    // Apply the prox. No lazy-updating here yet
    prox->call(iterate, step_t, iterate);
  }
}

inline double SGDAlt::get_step_t() {
  return step / (t + 1);
}
