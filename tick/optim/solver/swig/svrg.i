// License: BSD 3 clause

%include <std_shared_ptr.i>

%{
#include "svrg.h"
#include "model.h"
%}

class SVRG : public StoSolver {

public:
    enum class VarianceReductionMethod {
        Last    = 1,
        Average = 2,
        Random  = 3
    };

    enum class DelayedUpdatesMethod {
      Exact = 1,
      Proba = 2,
    };

    SVRG(unsigned long epoch_size,
         double tol,
         RandType rand_type,
         double step,
         int seed,
         VarianceReductionMethod variance_reduction = VarianceReductionMethod::Last,
         DelayedUpdatesMethod delayed_updates = DelayedUpdatesMethod::Exact);

    void solve();

    void set_step(double step);

    VarianceReductionMethod get_variance_reduction();

    DelayedUpdatesMethod get_delayed_updates();

    void set_variance_reduction(VarianceReductionMethod variance_reduction);

    void set_delayed_updates(DelayedUpdatesMethod delayed_updates);
};
