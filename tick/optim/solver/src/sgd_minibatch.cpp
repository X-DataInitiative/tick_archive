//
// Created by Martin Bompaire on 22/10/15.
//

#include <tick/optim/solver/src/sgd_minibatch.h>

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

    ulong k = 0;

    std::vector<ArrayDouble> local_sum_grads{};

    const ulong minibatch_size = 10;
    const ulong num_batches = epoch_size / minibatch_size;

    ulong total_t = t;
    #pragma omp parallel num_threads(4)
    {
      const auto num_threads = omp_get_num_threads();
      const auto thread_num = omp_get_thread_num();

      #pragma omp single
      local_sum_grads.resize(num_threads);

      local_sum_grads[thread_num] = ArrayDouble(iterate.size());

      ArrayDouble local_grad = ArrayDouble(iterate.size());
      ulong local_next_k;
      const ulong batches_per_thread = num_batches / num_threads;
      for (ulong batch_i = 0; batch_i < batches_per_thread; ++batch_i) {

        local_sum_grads[thread_num].fill(1.0);

        for (ulong j = 0; j < minibatch_size; ++j) {

          #pragma omp atomic capture
          local_next_k = k++;

          const ulong rand_index = rand_indices[local_next_k];
          model->grad_i(rand_index, iterate, local_grad);

//          #pragma omp simd
          for (ulong j = 0; j < iterate.size(); ++j) {
            local_sum_grads[thread_num][j] *= local_grad[j];
          }
        }

        #pragma omp barrier

        #pragma omp single nowait
        {
          for (ulong thread_i = 0; thread_i < num_threads; ++thread_i) {
            const auto local_step_t = get_step_t(t + (batch_i + thread_i) * minibatch_size);

            for (ulong j = 0; j < iterate.size(); ++j) {
              iterate[j] += local_sum_grads[thread_i][j] * 0.06;
              //prox->call(iterate, local_step_t, iterate);
            }

          }
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
