//
// Created by Martin Bompaire on 22/10/15.
//

#ifndef TICK_OPTIM_SOLVER_SRC_SDCA_H_
#define TICK_OPTIM_SOLVER_SRC_SDCA_H_

// License: BSD 3 clause

#include "model.h"
#include "../../prox/src/prox.h"
#include "sto_solver.h"


// TODO: profile the code of SDCA to check if it's faster

// TODO: code accelerated SDCA


class SDCA : public StoSolver {
 protected:
  ulong n_coeffs;

  // A boolean that attests that our arrays of ascent variables and dual variables are initialized
  // with the right size
  bool stored_variables_ready;

  // Store for coefficient update before prox call.
  ArrayDouble tmp_primal_vector;

  // Level of ridge regularization. This is mandatory for SDCA.
  double l_l2sq;

  // Ascent variables
  ArrayDouble delta;

  // The dual variable
  ArrayDouble dual_vector;

 public:
  explicit SDCA(double l_l2sq,
       ulong epoch_size = 0,
       double tol = 0.,
       RandType rand_type = RandType::unif,
       int seed = -1);

  void reset() override;

  void solve() override;

  void set_model(ModelPtr model) override;

  void init_stored_variables();
  void init_stored_variables(ArrayDouble &primal_vector, ArrayDouble &dual_vector);

  double get_l_l2sq() const {
    return l_l2sq;
  }

  void set_l_l2sq(double l_l2sq) {
    this->l_l2sq = l_l2sq;
  }

  SArrayDoublePtr get_primal_vector() const {
    ArrayDouble copy = tmp_primal_vector;
    return copy.as_sarray_ptr();
  }

  SArrayDoublePtr get_dual_vector() const {
    ArrayDouble copy = dual_vector;
    return copy.as_sarray_ptr();
  }

  void set_starting_iterate(ArrayDouble &new_iterate) override {
    TICK_ERROR("Cannot call set_starting_iterate in SDCA, use init_stored_variables")
  }
};

#endif  // TICK_OPTIM_SOLVER_SRC_SDCA_H_
