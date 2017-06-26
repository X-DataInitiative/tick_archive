//
#ifndef TICK_OPTIM_SOLVER_SRC_SGD_MINIBATCH_H_
#define TICK_OPTIM_SOLVER_SRC_SGD_MINIBATCH_H_

#include "model.h"
#include "../../prox/src/prox.h"
#include "sto_solver.h"

#include "model.h"
#include "../../prox/src/prox.h"
#include "sto_solver.h"

class SGDMinibatch : public StoSolver {
 private:
    double step_t;
    double step;

 public:
  SGDMinibatch(ulong epoch_size = 0,
        double tol = 0.,
        RandType rand_type = RandType::unif,
        double step = 0.,
        int seed = -1);

    inline double get_step_t() const {
        return step_t;
    }

    inline double get_step() const {
        return step;
    }

    inline void set_step(double step) {
        this->step = step;
    }

    void solve();

    void solve_sparse();

    inline double get_step_t(const ulong t);
};

#endif  // TICK_OPTIM_SOLVER_SRC_SGD_MINIBATCH_H_
