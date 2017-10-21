//
// Created by Stéphane GAIFFAS on 12/04/2016.
//

#ifndef TICK_OPTIM_MODEL_SRC_COXREG_FULL_LIK_H_
#define TICK_OPTIM_MODEL_SRC_COXREG_FULL_LIK_H_

// License: BSD 3 clause

#include "model_generalized_linear.h"

enum class BaselineType {
  exponential = 0,
  weibull,
  histogram
};

class DLL_PUBLIC ModelCoxRegFullLik : public ModelGeneralizedLinear {
 private:
  BaselineType baseline;

  // ulong n_samples, n_features, n_failures;

  // Number of bins used whenever baseline="histogram"
  uint32_t n_bins;

  SArrayUShortPtr censoring;

  inline double get_time(ulong i) const {
    return get_label(i);
  }

  inline ushort get_censoring(ulong i) const {
    return (*censoring)[i];
  }

 public:
  ModelCoxRegFullLik(const SBaseArrayDouble2dPtr features,
                     const SArrayDoublePtr times,
                     const SArrayUShortPtr censoring,
                     const BaselineType baseline,
                     const int n_threads);

  const char *get_class_name() const override {
    return "ModelCoxRegFullLik";
  }

  void grad_i(const ulong i, const ArrayDouble &coeffs, ArrayDouble &out) override;

  /**
   * To be used by grad(ArrayDouble&, ArrayDouble&) to calculate grad by incrementally
   * updating 'out'
   * out and coeffs are not in the same order as in grad_i as this is necessary for
   * parallel_map_array
   */
  virtual void inc_grad_i(const ulong i, ArrayDouble &out, const ArrayDouble &coeffs) override;

  void grad(const ArrayDouble &coeffs, ArrayDouble &out) override;

  double loss(const ArrayDouble &coeffs) override;

  bool use_intercept() const override {
    return fit_intercept;
  }

  bool is_sparse() const override {
    return features->is_sparse();
  }

  ulong get_n_coeffs() const override {
    TICK_CLASS_DOES_NOT_IMPLEMENT("get_n_coeffs");
  }

  void set_baseline(BaselineType baseline) {
    this->baseline = baseline;
  }

  inline BaselineType get_baseline() const {
    return baseline;
  }

  template<class Archive>
  void serialize(Archive &ar) {
    ar(cereal::make_nvp("ModelGeneralizedLinear", cereal::base_class<ModelGeneralizedLinear>(this)));
  }
};

CEREAL_SPECIALIZE_FOR_ALL_ARCHIVES(ModelCoxRegFullLik, cereal::specialization::member_serialize)

#endif  // TICK_OPTIM_MODEL_SRC_COXREG_FULL_LIK_H_
