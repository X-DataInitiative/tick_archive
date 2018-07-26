#include <chrono>

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


double StandardDeviation(std::vector<double>);
double Variance(std::vector<double>);

const constexpr size_t SEED       = 1933;
const constexpr size_t N_ITER     = 20;

// constexpr ulong n_samples = 196000;
constexpr ulong n_samples = 20000;

constexpr auto ALPHA = 100. / n_samples;
constexpr auto BETA  = 1e-10;
constexpr auto STRENGTH = ALPHA + BETA;
constexpr auto RATIO = BETA / STRENGTH;

int main(int argc, char *argv[]) {

  {
    std::string file_path = __FILE__;
    std::string dir_path = file_path.substr(0, file_path.rfind("/"));

    std::string features_s(dir_path + "/../data/adult.features.cereal");
    std::string labels_s(dir_path + "/../data/adult.labels.cereal");

    std::cout << "features_s=" << features_s << std::endl;
#ifdef _MKN_WITH_MKN_KUL_
    kul::File features_f(features_s);
    kul::File labels_f(labels_s);
    if(!features_f){
      features_s = "url.3.features.cereal";
      labels_s = "url.3.labels.cereal";
    }
#endif
    std::cout << "read features from " << features_s << std::endl;
    auto features(tick_double_sparse2d_from_file(features_s));
    std::cout << "features ("  << features->n_rows() << ", " << features->n_cols() << ")" << std::endl;

    std::cout << "read labels from " << labels_s << std::endl;
    auto labels(tick_double_array_from_file(labels_s));
    std::cout << "labels ("  << labels->size() << ", )" << std::endl;
    using milli = std::chrono::microseconds;
    std::vector<double> samples;
    for (int tries = 0; tries < 5; ++tries)
    {
      auto model = std::make_shared<ModelLogReg>(features, labels, false);
      Array<double> minimizer(model->get_n_coeffs());
      TSAGA<double> saga(n_samples, 0, RandType::unif, 0.00257480411965);
      saga.set_rand_max(n_samples);
      saga.set_model(model);
      auto prox = std::make_shared<TProxElasticNet<double, double>>(STRENGTH, RATIO, 0, model->get_n_coeffs(), 0);
      saga.set_prox(prox);
      size_t total = 0;
      auto start = NOW;

      for (int j = 0; j < N_ITER; ++j) {
        auto start_iter = NOW;
        saga.solve();
        total += (NOW - start_iter);
        saga.get_minimizer(minimizer);
        double objective = model->loss(minimizer) + prox->value(minimizer, prox->get_start(), prox->get_end());
        std::cout << "LogReg : " << j << " : time : " << total << " : objective: " << objective << std::endl;
      }
      auto finish = NOW;
      std::cout << argv[0] << " with n_threads " << std::to_string(1) << " "
                << total / 1e3
                << std::endl;
      samples.push_back(total / 1e3);
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