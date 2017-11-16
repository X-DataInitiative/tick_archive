//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef TICK_OPTIM_MODEL_SRC_POISREG_H_
#define TICK_OPTIM_MODEL_SRC_POISREG_H_

// License: BSD 3 clause

#include "model_generalized_linear.h"

// TODO: labels should be a ArrayUInt

enum class LinkType {
  identity = 0,
  exponential
};

class DLL_PUBLIC ModelPoisReg : public ModelGeneralizedLinear {
 private:
  LinkType link_type;
  bool ready_non_zero_label_map;
  VArrayULongPtr non_zero_labels;
  ulong n_non_zeros_labels;

 public:
  ModelPoisReg(const SBaseArrayDouble2dPtr features,
               const SArrayDoublePtr labels,
               const LinkType link_type,
               const bool fit_intercept,
               const int n_threads = 1);

  const char *get_class_name() const override {
    return "ModelPoisReg";
  }

  double loss_i(const ulong i, const ArrayDouble &coeffs) override;

  double grad_i_factor(const ulong i, const ArrayDouble &coeffs) override;

  SArrayDouble2dPtr hessian(ArrayDouble &coeffs);

  double sdca_dual_min_i(const ulong i,
                         const double dual_i,
                         const ArrayDouble &primal_vector,
                         const double previous_delta_dual_i,
                         double l_l2sq) override;

  std::tuple<double, double> sdca_dual_min_ij(const ulong i, const ulong j,
                                              const double dual_i, const double dual_j,
                                              const ArrayDouble &primal_vector,
                                              double l_l2sq) override;

  ArrayDouble sdca_dual_min_many(const ArrayULong indices,
                                 const ArrayDouble duals,
                                 const ArrayDouble &primal_vector,
                                 double l_l2sq) override;

  void sdca_primal_dual_relation(const double l_l2sq,
                                 const ArrayDouble &dual_vector,
                                 ArrayDouble &out_primal_vector) override;

  /**
   * Returns a mapping from the sampled observation (in [0, rand_max)) to the observation
   * position (in [0, n_samples)). For identity link this is needed as zero labeled observations
   * are discarded in SDCA.
   * For exponential link nullptr is returned, it means no index_map is required as the mapping
   * is the canonical inedx_map[i] = i
   */
  SArrayULongPtr get_sdca_index_map() override {
    if (link_type == LinkType::exponential) {
      return nullptr;
    }
    if (!ready_non_zero_label_map) init_non_zero_label_map();
    return non_zero_labels;
  }

 private:
  double sdca_dual_min_i_exponential(const ulong i,
                                     const double dual_i,
                                     const ArrayDouble &primal_vector,
                                     const double previous_delta_dual_i,
                                     double l_l2sq);
  /**
   * @brief Initialize the hash map that allow fast retrieving for get_non_zero_i
   */
  void init_non_zero_label_map();

  double sdca_dual_min_i_identity(const ulong i,
                                  const double dual_i,
                                  const ArrayDouble &primal_vector,
                                  const double previous_delta_dual_i,
                                  double l_l2sq);

  void compute_features_dot_products_ij(ulong i, ulong j, double _1_lambda_n,
                                        bool use_intercept,
                                        double &g_ii, double &g_jj, double &g_ij);

  void compute_primal_dot_products_ij(ulong i, ulong j, const ArrayDouble &primal_vector,
                                      double &p_i, double &p_j);

  void fill_gradient_ij(double label_i, double label_j,
                        double new_dual_i, double new_dual_j,
                        double delta_dual_i, double delta_dual_j,
                        double p_i, double p_j,
                        double g_ii, double g_jj, double g_ij,
                        double &n_grad_i, double &n_grad_j);

  void fill_hessian_ij(double label_i, double label_j,
                       double new_dual_i, double new_dual_j,
                       double g_ii, double g_jj, double g_ij,
                       double &n_hess_ii, double &n_hess_jj, double &n_hess_ij);

  void compute_descent(double n_grad_i, double n_grad_j,
                       double n_hess_ii, double n_hess_jj, double n_hess_ij,
                       double &newton_descent_i, double &newton_descent_j);

  void compute_features_dot_products_many(
    ulong n_indices, const ArrayULong &indices, double _1_lambda_n, bool use_intercept,
    ArrayDouble2d &g);

  void compute_primal_dot_products_many(
    ulong n_indices, const ArrayULong &indices, const ArrayDouble &primal_vector,
    ArrayDouble &p);

  void fill_gradient_hessian_many(
    ulong n_indices, ArrayDouble &labels, ArrayDouble &new_duals,
    ArrayDouble &delta_duals, ArrayDouble &p, ArrayDouble2d &g,
    ArrayDouble &n_grad, ArrayDouble2d & n_hess);

  void compute_descent_many(ulong n_indices, ArrayDouble &n_grad, ArrayDouble2d &n_hess);

 public:
  virtual void set_link_type(const LinkType link_type) {
    this->link_type = link_type;
  }

  virtual LinkType get_link_type() {
    return link_type;
  }
};

#endif  // TICK_OPTIM_MODEL_SRC_POISREG_H_
