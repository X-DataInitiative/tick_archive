// License: BSD 3 clause


#include "hawkes_fixed_sumexpkern_loglik.h"

ModelHawkesFixedSumExpKernLogLik::ModelHawkesFixedSumExpKernLogLik() :
  ModelHawkesSingle(), decays(0) {}

ModelHawkesFixedSumExpKernLogLik::ModelHawkesFixedSumExpKernLogLik(
  const ArrayDouble &decays, const int max_n_threads) :
  ModelHawkesSingle(max_n_threads, 0),
  decays(decays) {}

void ModelHawkesFixedSumExpKernLogLik::compute_weights() {
  allocate_weights();
  parallel_run(get_n_threads(),
               n_nodes,
               &ModelHawkesFixedSumExpKernLogLik::compute_weights_dim_i,
               this);
  weights_computed = true;

//  for (ulong k = 0; k < n_nodes; ++k) {
//    std::cout << "g " << k
//              << ", min g(k)=" << g[k].min()
//              << ", max g(k)=" << g[k].max()
//              << std::endl;
//    g[k].print();
//  }
//
  for (ulong k = 0; k < n_nodes; ++k) {
    std::cout << "\nG " << k
              << ", min G(k)=" << G[k].min()
              << ", max G(k)=" << G[k].max()
              << std::endl;
    G[k].print();
  }

  for (ulong k = 0; k < n_nodes; ++k) {
    std::cout << "\nsum_G " << k
              << ", min sum_G(k)=" << sum_G[k].min()
              << ", max sum_G(k)=" << sum_G[k].max()
              << std::endl;
    sum_G[k].print();
  }
}

void ModelHawkesFixedSumExpKernLogLik::allocate_weights() {
  if (n_nodes == 0) {
    TICK_ERROR("Please provide valid timestamps before allocating weights")
  }
  g = ArrayDouble2dList1D(n_nodes);
  G = ArrayDouble2dList1D(n_nodes);
  sum_G = ArrayDoubleList1D(n_nodes);

  for (ulong i = 0; i < n_nodes; i++) {
    g[i] = ArrayDouble2d((*n_jumps_per_node)[i], n_nodes * get_n_decays());
    g[i].init_to_zero();
    G[i] = ArrayDouble2d((*n_jumps_per_node)[i] + 1, n_nodes * get_n_decays());
    G[i].init_to_zero();
    sum_G[i] = ArrayDouble(n_nodes * get_n_decays());
  }
}

void ModelHawkesFixedSumExpKernLogLik::compute_weights_dim_i(const ulong i) {
  const ArrayDouble t_i = view(*timestamps[i]);
  ArrayDouble2d g_i = view(g[i]);
  ArrayDouble2d G_i = view(G[i]);
  ArrayDouble sum_G_i = view(sum_G[i]);

  const ulong n_jumps_i = (*n_jumps_per_node)[i];

  auto get_index = [=](ulong k, ulong j, ulong u) {
    return n_nodes * get_n_decays() * k + get_n_decays() * j + u;
  };

//  for (ulong k = 0; k < n_jumps_i + 1; k++) {
//    for (ulong j = 0; j < n_nodes; j++) {
//      for (ulong u = 0; u < get_n_decays(); ++u) {
//       std::cout << "get_index("<< k << ", " << j << ", "<< u << ")=" << get_index(k, j, u) << std::endl;
//      }
//    }
//  }
  for (ulong j = 0; j < n_nodes; j++) {
    const ArrayDouble t_j = view(*timestamps[j]);
    ulong ij = 0;
    for (ulong k = 0; k < n_jumps_i + 1; k++) {
      const double t_i_k = k < n_jumps_i ? t_i[k] : end_time;

      for (ulong u = 0; u < get_n_decays(); ++u) {
        if (k > 0) {
          const double ebt = std::exp(-decays[u] * (t_i_k - t_i[k - 1]));

          if (k < n_jumps_i) g_i[get_index(k, j, u)] = g_i[get_index(k - 1, j, u)] * ebt;
          G_i[get_index(k, j, u)] = g_i[get_index(k - 1, j, u)] * (1 - ebt) / decays[u];
        } else {
          g_i[get_index(k, j, u)] = 0;
          G_i[get_index(k, j, u)] = 0;
          sum_G_i[j * get_n_decays() + u] = 0.;
        }

        while ((ij < (*n_jumps_per_node)[j]) && (t_j[ij] < t_i_k)) {
          const double ebt = std::exp(-decays[u] * (t_i_k - t_j[ij]));
          if (k < n_jumps_i) g_i[get_index(k, j, u)] += decays[u] * ebt;
          G_i[get_index(k, j, u)] += 1 - ebt;
          ij++;
        }
        sum_G_i[j * get_n_decays() + u] += G_i[get_index(k, j, u)];
        std::cout << "i=" << i << ", j * get_n_decays() + u=" << j * get_n_decays() + u
//                  << ", adds : " << G_i[get_index(k, j, u)]
                  << "sum_G_i[j * get_n_decays() + u]=" << sum_G_i[j * get_n_decays() + u]
                  << std::endl;
      }
    }
  }
}

double ModelHawkesFixedSumExpKernLogLik::loss(const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();

  const double loss =
    parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                 &ModelHawkesFixedSumExpKernLogLik::loss_dim_i,
                                 this,
                                 coeffs);
  return loss / n_total_jumps;
}

double ModelHawkesFixedSumExpKernLogLik::loss_i(const ulong sampled_i,
                                                const ArrayDouble &coeffs) {
  if (!weights_computed) compute_weights();
  ulong i;
  ulong k;
  sampled_i_to_index(sampled_i, &i, &k);

  return loss_i_k(i, k, coeffs);
}

void ModelHawkesFixedSumExpKernLogLik::grad(const ArrayDouble &coeffs,
                                            ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  out.fill(0);

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(),
               n_nodes,
               &ModelHawkesFixedSumExpKernLogLik::grad_dim_i,
               this,
               coeffs,
               out);
  out /= n_total_jumps;
}

void ModelHawkesFixedSumExpKernLogLik::grad_i(const ulong sampled_i,
                                              const ArrayDouble &coeffs,
                                              ArrayDouble &out) {
  if (!weights_computed) compute_weights();

  ulong i;
  ulong k;
  sampled_i_to_index(sampled_i, &i, &k);

  // set grad to zero
  out.fill(0);

  grad_i_k(i, k, coeffs, out);
}

double ModelHawkesFixedSumExpKernLogLik::loss_and_grad(const ArrayDouble &coeffs,
                                                       ArrayDouble &out) {
  if (!weights_computed) compute_weights();
  out.fill(0);

  const double loss =
    parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                 &ModelHawkesFixedSumExpKernLogLik::loss_and_grad_dim_i,
                                 this,
                                 coeffs, out);
  out /= n_total_jumps;
  return loss / n_total_jumps;
}

double ModelHawkesFixedSumExpKernLogLik::hessian_norm(const ArrayDouble &coeffs,
                                                      const ArrayDouble &vector) {
  if (!weights_computed) compute_weights();

  const double norm_sum =
    parallel_map_additive_reduce(get_n_threads(), n_nodes,
                                 &ModelHawkesFixedSumExpKernLogLik::hessian_norm_dim_i,
                                 this,
                                 coeffs, vector);

  return norm_sum / n_total_jumps;
}

void ModelHawkesFixedSumExpKernLogLik::hessian(const ArrayDouble &coeffs, ArrayDouble &out) {
  if (!weights_computed) compute_weights();

  // This allows to run in a multithreaded environment the computation of each component
  parallel_run(get_n_threads(), n_nodes, &ModelHawkesFixedSumExpKernLogLik::hessian_i,
               this, coeffs, out);
  out /= n_total_jumps;
}



////////////////////////////////////////////////////////////////////////////////////////////////////
//                                    PRIVATE METHODS
////////////////////////////////////////////////////////////////////////////////////////////////////

void ModelHawkesFixedSumExpKernLogLik::sampled_i_to_index(const ulong sampled_i,
                                                          ulong *i,
                                                          ulong *k) {
  ulong cum_N_i = 0;
  for (ulong d = 0; d < n_nodes; d++) {
    cum_N_i += (*n_jumps_per_node)[d];
    if (sampled_i < cum_N_i) {
      *i = d;
      *k = sampled_i - cum_N_i + (*n_jumps_per_node)[d];
      break;
    }
  }
}

double ModelHawkesFixedSumExpKernLogLik::loss_dim_i(const ulong i,
                                                    const ArrayDouble &coeffs) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double loss = 0;
  loss += end_time * mu_i;

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; ++k) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);
    if (s <= 0) {
      TICK_ERROR("The sum of the influence on someone cannot be negative. "
                   "Maybe did you forget to add a positive constraint to "
                   "your proximal operator");
    }
    loss -= log(s);
  }

  loss += alpha_i.dot(sum_G[i]);
  return loss;
}

double ModelHawkesFixedSumExpKernLogLik::loss_i_k(const ulong i,
                                                  const ulong k,
                                                  const ArrayDouble &coeffs) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));
  double loss = 0;

  const ArrayDouble g_i_k = view_row(g[i], k);
  const ArrayDouble G_i_k = view_row(G[i], k);

  // Both are correct, just a question of point of view
  const double t_i_k = k == (*n_jumps_per_node)[i] - 1 ? end_time : (*timestamps[i])[k];
  const double t_i_k_minus_one = k == 0 ? 0 : (*timestamps[i])[k - 1];
  loss += (t_i_k - t_i_k_minus_one) * mu_i;
  //  loss += end_time * mu[i] / (*n_jumps_per_node)[i];

  double s = mu_i;
  s += alpha_i.dot(g_i_k);

  if (s <= 0) {
    TICK_ERROR("The sum of the influence on someone cannot be negative. Maybe did "
                 "you forget to add a positive constraint to your "
                 "proximal operator");
  }
  loss -= log(s);

  loss += alpha_i.dot(G_i_k);
  if (k == (*n_jumps_per_node)[i] - 1)
    loss += alpha_i.dot(view_row(G[i], k + 1));

  return loss;
}

void ModelHawkesFixedSumExpKernLogLik::grad_dim_i(const ulong i,
                                                  const ArrayDouble &coeffs,
                                                  ArrayDouble &out) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double &grad_mu_i = out[i];
  ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  grad_mu_i += end_time;

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; ++k) {
    const ArrayDouble g_i_k = view_row(g[i], k);
    double s = mu_i;
    s += alpha_i.dot(g_i_k);

    grad_mu_i -= 1. / s;
    grad_alpha_i.mult_incr(g_i_k, -1. / s);
  }

  grad_alpha_i.mult_incr(sum_G[i], 1);
}

void ModelHawkesFixedSumExpKernLogLik::grad_i_k(const ulong i, const ulong k,
                                                const ArrayDouble &coeffs,
                                                ArrayDouble &out) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double &grad_mu_i = out[i];
  ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  const ArrayDouble g_i_k = view_row(g[i], k);
  const ArrayDouble G_i_k = view_row(G[i], k);

  // Both are correct, just a question of point of view
  const double t_i_k = k == (*n_jumps_per_node)[i] - 1 ? end_time : (*timestamps[i])[k];
  const double t_i_k_minus_one = k == 0 ? 0 : (*timestamps[i])[k - 1];
  grad_mu_i += t_i_k - t_i_k_minus_one;
  //  grad_mu[i] += end_time / (*n_jumps_per_node)[i];

  double s = mu_i;
  s += alpha_i.dot(g_i_k);

  grad_mu_i -= 1. / s;
  grad_alpha_i.mult_incr(g_i_k, -1. / s);
  grad_alpha_i.mult_incr(G_i_k, 1.);

  if (k == (*n_jumps_per_node)[i] - 1)
    grad_alpha_i.mult_incr(view_row(G[i], k + 1), 1.);
}

double ModelHawkesFixedSumExpKernLogLik::loss_and_grad_dim_i(const ulong i,
                                                             const ArrayDouble &coeffs,
                                                             ArrayDouble &out) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double &grad_mu_i = out[i];
  ArrayDouble grad_alpha_i = view(out, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double loss = 0;

  grad_mu_i += end_time;
  loss += end_time * mu_i;
  for (ulong k = 0; k < (*n_jumps_per_node)[i]; k++) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);

    if (s <= 0) {
      TICK_ERROR("The sum of the influence on someone cannot be negative. Maybe did "
                   "you forget to add a positive constraint to your "
                   "proximal operator");
    }
    loss -= log(s);
    grad_mu_i -= 1. / s;

    grad_alpha_i.mult_incr(g_i_k, -1. / s);
  }

  loss += alpha_i.dot(sum_G[i]);
  grad_alpha_i.mult_incr(sum_G[i], 1);

  return loss;
}

double ModelHawkesFixedSumExpKernLogLik::hessian_norm_dim_i(const ulong i,
                                                            const ArrayDouble &coeffs,
                                                            const ArrayDouble &vector) {
  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double d_mu_i = vector[i];
  ArrayDouble d_alpha_i = view(vector, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  double hess_norm = 0;

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; k++) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double S = d_mu_i;
    S += d_alpha_i.dot(g_i_k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);

    double tmp = S / s;
    hess_norm += tmp * tmp;
  }
  return hess_norm;
}

void ModelHawkesFixedSumExpKernLogLik::hessian_i(const ulong i,
                                                 const ArrayDouble &coeffs,
                                                 ArrayDouble &out) {
  if (!weights_computed) TICK_ERROR("Please compute weights before calling hessian_i");

  const double mu_i = coeffs[i];
  const ArrayDouble alpha_i = view(coeffs, get_alpha_i_first_index(i), get_alpha_i_last_index(i));

  const ulong start_mu_line = i * (n_nodes + 1);
  const ulong block_start = (i + 1) * n_nodes * (n_nodes + 1);

  for (ulong k = 0; k < (*n_jumps_per_node)[i]; ++k) {
    const ArrayDouble g_i_k = view_row(g[i], k);

    double s = mu_i;
    s += alpha_i.dot(g_i_k);
    const double s_2 = s * s;

    // fill mu mu
    out[start_mu_line] += 1. / s_2;
    // fill mu alpha
    for (ulong j = 0; j < n_nodes; ++j) {
      out[start_mu_line + j + 1] += g_i_k[j] / s_2;
    }

    for (ulong l = 0; l < n_nodes; ++l) {
      const ulong start_alpha_line = block_start + l * (n_nodes + 1);
      // fill alpha mu
      out[start_alpha_line] += g_i_k[l] / s_2;
      // fill alpha square
      for (ulong m = 0; m < n_nodes; ++m) {
        out[start_alpha_line + m + 1] += g_i_k[l] * g_i_k[m] / s_2;
      }
    }
  }
}

ulong ModelHawkesFixedSumExpKernLogLik::get_n_coeffs() const {
  return n_nodes + n_nodes * n_nodes * get_n_decays();
}
