%include <std_shared_ptr.i>

%{
#include "sgd_minibatch.h"
#include "model.h"
%}

class SGDMinibatch : public StoSolver {

public:

    SGDMinibatch(unsigned long epoch_size,
        double tol,
        RandType rand_type,
        double step,
        int seed);

    inline void set_step(double step);

    inline double get_step() const;

    void solve();
};
