//
// Created by Martin Bompaire on 22/10/15.
//

#include "sgd_hogwild.h"

SGDHogwild::SGDHogwild(ulong epoch_size,
                       double tol,
                       RandType
                       rand_type,
                       double step,
                       int seed
)
    :
    StoSolver(epoch_size, tol, rand_type, seed
    ),
    step(step) {}

void SGDHogwild::solve() {
  if (model->is_sparse()) {
    solve_sparse();
  } else {
    // Dense case
    ArrayDouble grad(iterate.size());
    grad.init_to_zero();

    ArrayULong rand_indices(epoch_size);

    for (ulong i = 0; i < rand_indices.size(); ++i) {
      rand_indices[i] = get_next_i();
    }

    const ulong start_t = this->t;
    #pragma omp parallel num_threads(2)
    {
      #pragma omp for
      for (ulong t = start_t; t < start_t + epoch_size; ++t) {
        model->grad_i(rand_indices[t - start_t], iterate, grad);
        step_t = get_step_t(t);

//        #pragma omp critical
        iterate.mult_incr(grad, -step_t);

//      prox->call(iterate, step_t, iterate);

      }

    }

    this->t += epoch_size;
  }
}

void SGDHogwild::solve_sparse() {
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
    double step_t = get_step_t(t);
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

inline double SGDHogwild::get_step_t(const ulong t) {
  return step / (t + 1);
}
