//
// Created by Martin Bompaire on 21/10/15.
//

#ifndef TICK_OPTIM_MODEL_SRC_POISREG_H_
#define TICK_OPTIM_MODEL_SRC_POISREG_H_

// License: BSD 3 clause

#include <unordered_map>

#include "model_generalized_linear.h"

// TODO: labels should be a ArrayUInt

enum class LinkType {
  identity = 0,
  exponential
};

class ModelPoisReg : public ModelGeneralizedLinear {
 private:
  LinkType link_type;
  bool non_zero_label_map_computed;
  std::unordered_map<ulong, ulong> non_zero_label_map;

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

  double sdca_dual_min_i(const ulong i,
                         const ArrayDouble &dual_vector,
                         const ArrayDouble &primal_vector,
                         const ArrayDouble &previous_delta_dual,
                         const double l_l2sq) override;

  BaseArrayDouble get_sdca_features(const ulong i) override;

 private:
  double sdca_dual_min_i_exponential(const ulong i,
                                     const ArrayDouble &dual_vector,
                                     const ArrayDouble &primal_vector,
                                     const ArrayDouble &previous_delta_dual,
                                     const double l_l2sq);

  ulong get_non_zero_i(ulong i);

  void init_non_zero_label_map();

  double sdca_dual_min_i_identity(const ulong i,
                                  const ArrayDouble &dual_vector,
                                  const ArrayDouble &primal_vector,
                                  const ArrayDouble &previous_delta_dual,
                                  const double l_l2sq);

 public:
  virtual void set_link_type(const LinkType link_type) {
    this->link_type = link_type;
  }
};

#endif  // TICK_OPTIM_MODEL_SRC_POISREG_H_
