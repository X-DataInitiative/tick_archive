//
// Created by Martin Bompaire on 22/10/15.
//

#include "sgd_minibatch.h"

#include <omp.h>

SGDMinibatch::SGDMinibatch(ulong epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed)
    : StoSolver(epoch_size, tol, rand_type, seed),
      step(step) {}

void SGDMinibatch::solve() {
  if (model->is_sparse()) {
    solve_sparse();
  } else {
    // Dense case

    ArrayULong rand_indices(epoch_size);
    for (ulong i = 0; i < rand_indices.size(); ++i)
      rand_indices[i] = get_next_i();

    std::vector<ArrayDouble> local_sum_grads{};

    const ulong minibatch_size = 4;
    const ulong num_batches = epoch_size / minibatch_size;

    ulong local_t = t;
    #pragma omp parallel num_threads(1)
    {
      const ulong num_threads = static_cast<ulong>(omp_get_num_threads());
      const ulong thread_num = static_cast<ulong>(omp_get_thread_num());

      #pragma omp single
      local_sum_grads.resize(num_threads);

      local_sum_grads[thread_num] = ArrayDouble(iterate.size());

      ArrayDouble local_grad = ArrayDouble(iterate.size());

      const ulong num_batches_per_thread = num_batches / num_threads;
      for (ulong batch_i = 0; batch_i < num_batches_per_thread; ++batch_i) {
        local_sum_grads[thread_num].init_to_zero();

        for (ulong j = 0; j < minibatch_size; ++j) {
          const ulong k = (thread_num * num_batches_per_thread * minibatch_size) + batch_i * minibatch_size + j;

          model->grad_i(rand_indices[k], iterate, local_grad);

          #pragma omp simd
          for (ulong f_i = 0; f_i < iterate.size(); ++f_i) {
            local_sum_grads[thread_num][f_i] += local_grad[f_i];
          }
        }

        #pragma omp simd
        for (ulong f_i = 0; f_i < iterate.size(); ++f_i) {
          local_sum_grads[thread_num][f_i] *= (1.0 / minibatch_size);
        }

        #pragma omp barrier

        #pragma omp single
        {
          ArrayDouble sum_grad(iterate.size());
          sum_grad.init_to_zero();

          for (ulong thread_i = 0; thread_i < num_threads; ++thread_i) {
//            const auto local_step_t = get_step_t(t + (num_threads * batch_i * minibatch_size) + thread_i * (minibatch_size / 2));
            const auto local_step_t = get_step_t(local_t);

            #pragma omp simd
            for (ulong f_i = 0; f_i < iterate.size(); ++f_i) {
              sum_grad[f_i] += local_sum_grads[thread_i][f_i] * (-local_step_t);
            }

            local_t += minibatch_size;
          }

          #pragma omp simd
          for (ulong f_i = 0; f_i < iterate.size(); ++f_i) {
            iterate[f_i] += (sum_grad[f_i] / num_threads);
          }

          prox->call(iterate, get_step_t(local_t), iterate);
        }
      }
    }

    t += epoch_size;
  }
}

void SGDMinibatch::solve_sparse() {
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
    double step_t = get_step_t(0);
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

inline double SGDMinibatch::get_step_t(const ulong t) {
  return step / (t + 1);
}
