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

#ifdef _MKN_WITH_MKN_KUL_
#include "kul/os.hpp"
#endif

#define NOW std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()

const constexpr int SEED          = 42;



double StandardDeviation(std::vector<double>);
double Variance(std::vector<double>);


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
  if(argc <= 1) range.push_back(2);
  else range.push_back(std::stoi(argv[1]));

  ulong n_iter;
  if (argc <= 2) n_iter = 25;
  else n_iter = std::stoul(argv[2]);

  int record_every;
  if (argc <= 3) record_every = 4;
  else record_every = std::stoul(argv[3]);

  auto features(tick_double_sparse2d_from_file(features_s));

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


    std::vector<double> samples;
    for (int tries = 0; tries < 5; ++tries)
    {
      auto model = std::make_shared<TModelLogReg<double, std::atomic<double> > >(features, labels, false);
      Array<std::atomic<double>> minimizer(model->get_n_coeffs());
      AtomicSAGA<double> saga(
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
      auto prox = std::make_shared<TProxElasticNet<double, std::atomic<double> >>(STRENGTH, RATIO, 0, model->get_n_coeffs(), 0);
      saga.set_prox(prox);
      saga.solve(); // single solve call as iterations happen within threads
      const auto &history = saga.get_history();
      const auto &objective = saga.get_objective();

      const auto min_objective = saga.get_objective().min();

      for(ulong i = 1; i < objective.size(); i++) {
        auto log_dist = objective[i] == min_objective? 0: log10(objective[i] - min_objective);
        std::cout << n_threads << " " << i * record_every << " " << history[i] << " "
                  << "1e" << log_dist <<  std::endl;
      }
      std::cout << "min_objective : " << min_objective << "\n" << std::endl;

      samples.push_back(history.last());
    }

    double min = *std::min_element(samples.begin(), samples.end());
    double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
    double ci_half_width = 1.96 * StandardDeviation(samples) / std::sqrt(samples.size());

    std::cout << "\n"
              << "Min: " << min
              << "\n Mean: " << mean << " +/- " << ci_half_width
              << std::endl;
  }

  return 0;
}


double StandardDeviation(std::vector<double> samples)
{
  return std::sqrt(Variance(samples));
}

double Variance(std::vector<double> samples)
{
  int size = samples.size();

  double variance = 0;
  double t = samples[0];
  for (int i = 1; i < size; i++)
  {
    t += samples[i];
    double diff = ((i + 1) * samples[i]) - t;
    variance += (diff * diff) / ((i + 1.0) *i);
  }

  return variance / (size - 1);
}
