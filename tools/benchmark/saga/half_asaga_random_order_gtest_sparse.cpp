#include <chrono>
#include <iomanip>      // std::setprecision


#include "tick/array/serializer.h"
#include "tick/random/test_rand.h"
#include "tick/solver/saga.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_linreg.h"

#include "tick/prox/prox_zero.h"
#include "tick/prox/prox_elasticnet.h"


#include <math.h>
#include <vector>

#include "shared_saga.h"

#ifdef _MKN_WITH_MKN_KUL_
#include "kul/os.hpp"
#endif

#define NOW std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

const constexpr int SEED          = 42;

std::tuple<ArrayDouble, ArrayDouble> run_half_asaga_solver(
    SBaseArrayDouble2dPtr features, SArrayDoublePtr labels, ulong n_iter, int n_threads,
    int record_every, double strength, double ratio) {
  const auto n_samples = features->n_rows();

  auto model = std::make_shared<TModelLogReg<double> >(features, labels, false);
  HalfAtomicSAGA<double> saga(
      n_samples, n_iter,
      0,
      RandType::unif,
      1. / model->get_lip_max(),
      SEED,
      n_threads
  );
  saga.set_rand_max(n_samples);
  saga.set_model(model);

  saga.set_record_every(record_every);
  auto prox = std::make_shared<TProxElasticNet<double>>(strength, ratio, 0, model->get_n_coeffs(), 0);
  saga.set_prox(prox);
  saga.solve(); // single solve call as iterations happen within threads
  const auto &history = saga.get_history();
  const auto &objective = saga.get_objective();

  return std::make_tuple(history, objective);
}


int main(int argc, char *argv[]) {
  submain(argc, argv, run_half_asaga_solver);
  return 0;
}

