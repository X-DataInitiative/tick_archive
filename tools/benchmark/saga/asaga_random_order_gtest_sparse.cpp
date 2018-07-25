#include <chrono>

#include "tick/array/serializer.h"
#include "tick/random/test_rand.h"
#include "tick/solver/saga.h"
#include "tick/linear_model/model_logreg.h"
#include "tick/linear_model/model_linreg.h"

#include "tick/prox/prox_zero.h"
#include "tick/prox/prox_elasticnet.h"

#ifdef _MKN_WITH_MKN_KUL_
#include "kul/os.hpp"
#endif

#define NOW std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

const constexpr int SEED          = 42;
const constexpr size_t N_ITER     = 15;


int main(int argc, char *argv[]) {
  std::string file_path = __FILE__;
  std::string dir_path = file_path.substr(0, file_path.rfind("/"));

  std::string features_s(dir_path + "/../data/adult.features.cereal");
  std::string labels_s(dir_path + "/../data/adult.labels.cereal");
#ifdef _MKN_WITH_MKN_KUL_
  kul::File features_f(features_s);
  kul::File labels_f(labels_s);
  if(!features_f){
    features_s = "url.features.cereal";
    labels_s = "url.labels.cereal";
  }
#endif

  std::vector<int> range;//{ 12}; //, 4, 6, 8, 10, 12, 14, 16 };
  if(argc == 1) range.push_back(2);
  else range.push_back(std::stoi(argv[1]));


  auto features(tick_double_sparse2d_from_file(features_s));

  std::cout << "features.indices() "  << features->indices() << std::endl;
  std::cout << "features.indices()[-1] "  << features->indices()[features->size_sparse() - 1] << std::endl;

  std::cout << "features.n_rows() "  << features->n_rows() << std::endl;
  std::cout << "features.size_sparse() "  << features->size_sparse() << std::endl;
  auto labels(tick_double_array_from_file(labels_s));

  const auto n_samples = features->n_rows();
  const auto ALPHA = 100. / n_samples;
  const auto BETA  = 1e-10;
  const auto STRENGTH = ALPHA + BETA;
  const auto RATIO = BETA / STRENGTH;


  for(auto n_threads : range){

    // using milli = std::chrono::microseconds;
    {
      auto model = std::make_shared<TModelLogReg<double, std::atomic<double> > >(features, labels, false);
      Array<std::atomic<double>> minimizer(model->get_n_coeffs());
      AtomicSAGA<double> saga(
        n_samples, N_ITER / n_threads,
        0,
        RandType::unif,
        0.00257480411965, //1e-3,
        SEED,
        n_threads,
        SAGA_VarianceReductionMethod::Last
      );
      saga.set_rand_max(n_samples);
      saga.set_model(model);
      auto prox = std::make_shared<TProxElasticNet<double, std::atomic<double> >>(STRENGTH, RATIO, 0, model->get_n_coeffs(), 0);
      saga.set_prox(prox);
      saga.solve(); // single solve call as iterations happen within threads
      const auto &history = saga.get_history();
      const auto &objective = saga.get_objective();
      for(size_t i = 0; i < N_ITER / n_threads; i++)
        std::cout << n_threads << " " << (i * n_threads) << " " << history[i] << " " << objective[i] << std::endl;
    }
  }

  return 0;
}
