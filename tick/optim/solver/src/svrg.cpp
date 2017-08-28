// License: BSD 3 clause

#include <model_labels_features.h>
#include "svrg.h"

SVRG::SVRG(ulong epoch_size,
           double tol,
           RandType rand_type,
           double step,
           int seed,
           VarianceReductionMethod variance_reduction,
           DelayedUpdatesMethod delayed_updates
)
    : StoSolver(epoch_size, tol, rand_type, seed),
      step(step), variance_reduction(variance_reduction),
      delayed_updates(delayed_updates),
      ready_step_corrections(false) {}

void SVRG::set_model(ModelPtr model) {
  StoSolver::set_model(model);
  ready_step_corrections = false;
}

void SVRG::set_prox(ProxPtr prox) {
  StoSolver::set_prox(prox);
  if (prox->is_separable()) {
    casted_prox = std::static_pointer_cast<ProxSeparable>(prox);
  }
}

void SVRG::prepare_solve() {
  // The point where we compute the full gradient for variance reduction is the
  // new iterate obtained at the previous epoch
  fixed_w = next_iterate;
  // Allocation and computation of the full gradient
  full_gradient = ArrayDouble(iterate.size());
  model->grad(fixed_w, full_gradient);
  if ((model->is_sparse()) && (prox->is_separable())) {
    if (delayed_updates == DelayedUpdatesMethod::Exact) {
      // TODO: Initialize this only when model is setted ?
      last_time = ArrayULong(model->get_n_features());
      last_time.fill(0);
      if (!ready_step_corrections) {
        compute_step_corrections();
      }
    }
  } else {
    grad_i = ArrayDouble(iterate.size());
    grad_i_fixed_w = ArrayDouble(iterate.size());
  }
  rand_index = 0;
  if (variance_reduction == VarianceReductionMethod::Random ||
      variance_reduction == VarianceReductionMethod::Average) {
    next_iterate.init_to_zero();
  }
  if (variance_reduction == VarianceReductionMethod::Random) {
    rand_index = rand_unif(epoch_size);
  }
}

void SVRG::solve() {
  prepare_solve();
  if ((model->is_sparse()) && (prox->is_separable())) {
    bool use_intercept = model->use_intercept();
    ulong n_features = model->get_n_features();
    if (delayed_updates == DelayedUpdatesMethod::Exact) {
      solve_sparse_exact_updates(use_intercept, n_features);
    }
    if (delayed_updates == DelayedUpdatesMethod::Proba) {
      solve_sparse_proba_updates(use_intercept, n_features);
    }
  } else {
    solve_dense();
  }
}

void SVRG::compute_step_corrections() {
  if ((model->is_sparse()) && delayed_updates == DelayedUpdatesMethod::Proba) {
    ulong n_features = model->get_n_features();
    ulong n_samples = model->get_n_samples();
    std::shared_ptr<ModelLabelsFeatures> casted_model;
    casted_model = std::dynamic_pointer_cast<ModelLabelsFeatures>(model);
    ArrayULong columns_non_zeros(n_features);
    if (!casted_model->is_ready_columns_sparsity()) {
      casted_model->compute_columns_non_zeros(columns_non_zeros);
    }
    steps_correction = ArrayDouble(n_features);
    for (ulong j = 0; j < n_features; ++j) {
      steps_correction[j] = static_cast<double>(n_samples / columns_non_zeros[j]);
    }
    ready_step_corrections = true;
  }
}

void SVRG::solve_dense() {
  for (ulong t = 0; t < epoch_size; ++t) {
    ulong i = get_next_i();
    model->grad_i(i, iterate, grad_i);
    model->grad_i(i, fixed_w, grad_i_fixed_w);
    for (ulong j = 0; j < iterate.size(); ++j) {
      iterate[j] = iterate[j] - step * (grad_i[j] - grad_i_fixed_w[j] + full_gradient[j]);
    }
    prox->call(iterate, step, iterate);
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }
    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  if (variance_reduction == VarianceReductionMethod::Last) {
    next_iterate = iterate;
  }
  t += epoch_size;
}

void SVRG::solve_sparse_proba_updates(bool use_intercept, ulong n_features) {
  // Data is sparse, and we use the probabilistic update strategy
  // This means that the model is a child of ModelGeneralizedLinear.
  // The strategy used here uses non-delayed updates, with corrected
  // step-sizes using a probabilistic approximation and the
  // penalization trick: with such a model and prox, we can work only inside the current
  // support (non-zero values) of the sampled vector of features
  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();
    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);
    // Gradients factors (model is a GLM)
    // TODO: a grad_i_factor(i, array1, array2) to loop once on the features
    double grad_i_diff = model->grad_i_factor(i, iterate) - model->grad_i_factor(i, fixed_w);
    // We update the iterate within the support of the features vector, with the probabilistic correction
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];
      double full_gradient_j = full_gradient[j];
      // Step-size correction for coordinate j
      double step_correction = steps_correction[j];
      // Gradient descent with probabilistic step-size correction
      iterate[j] -= step * (x_i.data()[idx_nnz] * grad_i_diff + step_correction * full_gradient_j);
      // Prox is separable, apply regularization on the current coordinate
      iterate[j] = casted_prox->call_single(iterate[j], step * step_correction);
    }
    // And let's not forget to update the intercept as well
    // It's updated at each step, so no step-correction and no prox applied it
    if (use_intercept) {
      iterate[n_features] -= step * (grad_i_diff + full_gradient[n_features]);
    }
    // Note that the average option for variance reduction with sparse data is a very bad idea,
    // but this is catched in the python class
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }
    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  t += epoch_size;
  if (variance_reduction == VarianceReductionMethod::Last) {
    next_iterate = iterate;
  }
}

void SVRG::solve_sparse_exact_updates(bool use_intercept, ulong n_features) {
  // Data is sparse, and we use the exact strategy based on delayed updates.
  // This means that the model is a child of ModelGeneralizedLinear.
  // The strategy used here uses delayed updates strategy and the
  // delayed penalization trick when the prox is separable, so that
  // we can work when possible only inside the current support
  // (non-zero values) of the sampled vector of features
  for (t = 0; t < epoch_size; ++t) {
    // Get next sample index
    ulong i = get_next_i();
    // Sparse features vector
    BaseArrayDouble x_i = model->get_features(i);
    // Gradients factors (model is a GLM)
    // TODO: a grad_i_factor(i, array1, array2) to loop once on the features
    double grad_i_diff = model->grad_i_factor(i, iterate) - model->grad_i_factor(i, fixed_w);
    // We update the iterate within the support of the features vector
    for (ulong idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      ulong j = x_i.indices()[idx_nnz];
      // How many iterations since the last update of feature j ?
      ulong delay_j = 0;
      ulong last_time_j_plus_one = last_time[j] + 1;
      if (t > last_time_j_plus_one) {
        delay_j = t - last_time_j_plus_one;
      }
      // Full gradient's coordinate j
      double full_gradient_j = full_gradient[j];
      // If there is delay
      if (delay_j > 0) {
        // then we need to correct variance reduction
        iterate[j] -= step * delay_j * full_gradient_j;
        iterate[j] = casted_prox->call_single(iterate[j], step, delay_j);
      }
      // Apply gradient descent to the model weights in the support of x_i
      iterate[j] -= step * (x_i.data()[idx_nnz] * grad_i_diff + full_gradient_j);
      iterate[j] = casted_prox->call_single(iterate[j], step);
      // Update last_time
      last_time[j] = t;
    }
    // And let's not forget to update the intercept as well
    // It's updated at each step, so no step-correction and no prox applied it
    if (use_intercept) {
      iterate[n_features] -= step * (grad_i_diff + full_gradient[n_features]);
    }
    // Note that the average option for variance reduction with sparse data is a very bad idea,
    // but this is catched in the python class
    if (variance_reduction == VarianceReductionMethod::Random && t == rand_index) {
      next_iterate = iterate;
    }
    if (variance_reduction == VarianceReductionMethod::Average) {
      next_iterate.mult_incr(iterate, 1.0 / epoch_size);
    }
  }
  // Now we need to fully update the iterate (not the intercept),
  // since we reached the end of the epoch
  for (ulong j = 0; j < n_features; ++j) {
    ulong delay_j = 0;
    ulong last_time_j_plus_one = last_time[j] + 1;
    if (t > last_time_j_plus_one) {
      delay_j = t - last_time_j_plus_one;
    }
    if (delay_j > 0) {
      iterate[j] -= step * delay_j * full_gradient[j];
      iterate[j] = casted_prox->call_single(iterate[j], step, delay_j);
    }
  }
  t += epoch_size;
  if (variance_reduction == VarianceReductionMethod::Last) {
    next_iterate = iterate;
  }
}

void SVRG::set_starting_iterate(ArrayDouble &new_iterate) {
  StoSolver::set_starting_iterate(new_iterate);
  next_iterate = iterate;
}
