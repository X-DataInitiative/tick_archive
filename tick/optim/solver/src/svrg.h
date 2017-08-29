#ifndef TICK_OPTIM_SOLVER_SRC_SVRG_H_
#define TICK_OPTIM_SOLVER_SRC_SVRG_H_

// License: BSD 3 clause

#include "array.h"
#include "sgd.h"
#include "../../prox/src/prox.h"
#include "../../prox/src/prox_separable.h"

class SVRG : public StoSolver {
 public:
  enum class VarianceReductionMethod {
    Last = 1,
    Average = 2,
    Random = 3,
  };

  enum class DelayedUpdatesMethod {
    Exact = 1,
    Proba = 2,
  };

 private:
  double step;
  // Probabilistic correction of the step-sizes of all model weights,
  // given by the inverse proportion of non-zero entries in each feature column
  ArrayDouble steps_correction;

  VarianceReductionMethod variance_reduction;
  DelayedUpdatesMethod delayed_updates;

  ArrayDouble full_gradient;
  ArrayDouble fixed_w;
  ArrayDouble grad_i;
  ArrayDouble grad_i_fixed_w;
  ArrayDouble next_iterate;

  ulong rand_index;
  bool ready_step_corrections;

  // The array will contain the iteration index of the last update of each
  // coefficient (model-weights)
  ArrayULong last_time;

  void prepare_solve();

  void solve_dense();

  void solve_sparse_proba_updates(bool use_intercept, ulong n_features);

  void solve_sparse_exact_updates(bool use_intercept, ulong n_features);

  void compute_step_corrections();

 public:
  SVRG(ulong epoch_size,
       double tol,
       RandType rand_type,
       double step,
       int seed = -1,
       VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last,
       DelayedUpdatesMethod delayed_updates = DelayedUpdatesMethod::Exact);

  void solve() override;

  void set_model(ModelPtr model) override;

  void set_prox(ProxPtr prox) override;

  double get_step() const {
    return step;
  }

  void set_step(double step) {
    SVRG::step = step;
  }

  VarianceReductionMethod get_variance_reduction() const {
    return variance_reduction;
  }

  DelayedUpdatesMethod get_delayed_updates() const {
    return delayed_updates;
  }

  void set_variance_reduction(VarianceReductionMethod variance_reduction) {
    SVRG::variance_reduction = variance_reduction;
  }

  void set_delayed_updates(DelayedUpdatesMethod delayed_updates) {
    this->delayed_updates = delayed_updates;
  }

  void set_starting_iterate(ArrayDouble &new_iterate) override;
};

#endif  // TICK_OPTIM_SOLVER_SRC_SVRG_H_
