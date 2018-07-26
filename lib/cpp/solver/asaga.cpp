// License: BSD 3 clause

#include "tick/solver/saga.h"

template <class T>
AtomicSAGA<T>::AtomicSAGA(ulong epoch_size, ulong _iterations, T tol,
                          RandType rand_type, T step, int seed, int n_threads)
    : TBaseSAGA<T, std::atomic<T>>(epoch_size, tol, rand_type, step, seed),
      n_threads(n_threads),
      iterations(_iterations),
      objective(_iterations),
      history(_iterations) {
  un_threads = (size_t)n_threads;
}

template <class T>
void AtomicSAGA<T>::solve_dense(bool use_intercept, ulong n_features) {
  T n_samples = model->get_n_samples();

  auto lambda = [&]() {
    for (ulong t = 0; t < epoch_size; ++t) {
      // Get next sample index
      ulong i = get_next_i();
      // Get the features matrix. We know that it's dense
      BaseArray<T> x_i = model->get_features(i);
      T grad_i_factor = model->grad_i_factor(i, iterate);
      T grad_i_factor_old = gradients_memory[i];
      // Update gradient memory
      gradients_memory[i] = grad_i_factor;
      T grad_factor_diff = grad_i_factor - grad_i_factor_old;
      for (ulong j = 0; j < n_features; ++j) {
        T x_ij = x_i._value_dense(j);
        T grad_avg_j = gradients_average[j];

        // Update the gradients average over seen samples

        T iterate_j = iterate[j].load();
        while (!iterate[j].compare_exchange_weak(
            iterate_j,
            iterate_j - (step * (grad_factor_diff * x_ij + grad_avg_j)))) {
        }

        T gradients_average_j = gradients_average[j].load();
        while (!gradients_average[j].compare_exchange_weak(
            gradients_average_j,
            gradients_average_j + (grad_factor_diff * x_ij / n_samples))) {
        }
      }
      // deal with intercept here
      if (use_intercept) {
        T iterate_j = iterate[n_features];
        T gradients_average_j = gradients_average[n_features];
        while (!iterate[n_features].compare_exchange_weak(
            iterate_j,
            iterate_j - (step * (grad_factor_diff + gradients_average_j)))) {
        }

        while (!gradients_average[n_features].compare_exchange_weak(
            gradients_average_j,
            gradients_average_j + (grad_factor_diff / n_samples))) {
        }
      }
      // Call the prox on the iterate
      prox->call(iterate, step, iterate);
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < un_threads; i++) {
    threads.emplace_back(lambda);
  }
  for (size_t i = 0; i < un_threads; i++) {
    threads[i].join();
  }
  TStoSolver<T, std::atomic<T>>::t += epoch_size;
}

template <class T>
void AtomicSAGA<T>::solve_sparse_proba_updates(bool use_intercept,
                                               ulong n_features) {
  // Data is sparse, and we use the probabilistic update strategy
  // This means that the model is a child of ModelGeneralizedLinear.
  // The strategy used here uses non-delayed updates, with corrected
  // step-sizes using a probabilistic approximation and the
  // penalization trick: with such a model and prox, we can work only inside the
  // current support (non-zero values) of the sampled vector of features

  T n_samples = model->get_n_samples();
  T n_samples_inverse = 1 / n_samples;

  Array<std::atomic<T>> minimizer(model->get_n_coeffs());


  ulong n_records = std::ceil(static_cast<double>(iterations)/ record_every);
  history = ArrayDouble(n_records);
  iterates_history = ArrayDouble2d(n_records, model->get_n_coeffs());

  auto lambda = [&](uint16_t n_thread) {
    T grad_factor_diff = 0;
    T x_ij = 0;
    T grad_avg_j = 0;
    T step_correction = 0;
    T iterate_j = 0;
    T grad_i_factor = 0;
    T grad_i_factor_old = 0;

    struct timespec start, finish;
    T elapsed;
#if !defined(_WIN32)  // temporarily disabled TODO DOESN'T WORK ON WINDOWS
    clock_gettime(CLOCK_MONOTONIC, &start);
#endif
    ulong idx_nnz = 0;

    int thread_epoch_size = epoch_size / n_threads;

    for (ulong t = 0; t < thread_epoch_size * iterations; ++t) {
      // Get next sample index
      ulong i = get_next_i();
      // Sparse features vector
      BaseArray<T> x_i = model->get_features(i);
      grad_i_factor = model->grad_i_factor(i, iterate);
      grad_i_factor_old = gradients_memory[i].load();

      while (!gradients_memory[i].compare_exchange_weak(grad_i_factor_old, grad_i_factor));

      grad_factor_diff = grad_i_factor - grad_i_factor_old;
      for (idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
        // Get the index of the idx-th sparse feature of x_i
        ulong j = x_i.indices()[idx_nnz];
        x_ij = x_i.data()[idx_nnz];
        grad_avg_j = gradients_average[j].load();
        // Step-size correction for coordinate j
        step_correction = steps_correction[j];

        while (!gradients_average[j].compare_exchange_weak(
            grad_avg_j,
            grad_avg_j + (grad_factor_diff * x_ij * n_samples_inverse)));

        // Prox is separable, apply regularization on the current coordinate
        iterate[j] = casted_prox->call_single(
            iterate[j] - (step * (grad_factor_diff * x_ij +
                step_correction * grad_avg_j)),
            step * step_correction);

      }
      // And let's not forget to update the intercept as well. It's updated at
      // each step, so no step-correction. Note that we call the prox, in order
      // to be consistent with the dense case (in the case where the user has
      // the weird desire to to regularize the intercept)
      if (use_intercept) {
        T iterate_j = iterate[n_features];
        T gradients_average_j = gradients_average[n_features];
        while (!iterate[n_features].compare_exchange_weak(
            iterate_j,
            iterate_j - (step * (grad_factor_diff + gradients_average_j)))) {
        }

        while (!gradients_average[n_features].compare_exchange_weak(
            gradients_average_j,
            gradients_average_j + (grad_factor_diff / n_samples))) {
        }
        casted_prox->call_single(n_features, iterate, step, iterate);
      }

      if (n_thread == 0 && t % (thread_epoch_size * record_every) == 0) {
#if !defined(_WIN32)  // temporarily disabled TODO DOESN'T WORK ON WINDOWS
        int64_t index = t / (thread_epoch_size * record_every);
        clock_gettime(CLOCK_MONOTONIC, &finish);
        elapsed = (finish.tv_sec - start.tv_sec);
        elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
        history[index] = static_cast<double>(elapsed);
#endif
        for (ulong j = 0; j < iterate.size(); ++j) {
          iterates_history(index, j) = iterate[j].load();
        }
      }
    }
  };

  std::vector<std::thread> threads;
  for (size_t i = 0; i < un_threads; i++) {
    threads.emplace_back(lambda, i);
  }
  for (size_t i = 0; i < un_threads; i++) {
    threads[i].join();
  }

  objective = ArrayDouble(n_records);
  Array<std::atomic<T>> atomicIterate(iterate.size());
  for (ulong index = 0; index < n_records; ++index) {
    for (ulong j = 0; j < iterate.size(); ++j) {
      atomicIterate[j].store(view_row(iterates_history, index)[j]);
    }
    objective[index] = model->loss(atomicIterate) +
            prox->value(atomicIterate, prox->get_start(), prox->get_end());
  }

  TStoSolver<T, std::atomic<T>>::t += epoch_size;
}

template <class T>
void AtomicSAGA<T>::get_atomic_minimizer(Array<std::atomic<T>>& out) {
  for (ulong i = 0; i < iterate.size(); ++i) {
    out[i].store(iterate[i]);
  }
}

template class DLL_PUBLIC AtomicSAGA<double>;
template class DLL_PUBLIC AtomicSAGA<float>;
