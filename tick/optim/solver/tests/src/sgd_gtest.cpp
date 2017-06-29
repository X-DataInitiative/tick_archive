#include <random>

#include <gtest/gtest.h>

#include <tick/optim/model/src/linreg.h>
#include <tick/optim/solver/src/sgd.h>
#include <tick/optim/solver/src/sgd_minibatch.h>
#include <tick/optim/solver/src/sgd_alt.h>
#include <tick/optim/solver/src/sgd_hogwild.h>
#include <tick/optim/prox/src/prox_l2sq.h>

const ulong num_iterations = 2000;
const ulong n_features = 5;
const ulong n_samples = 10000;
const double initial_step_size = 50;

const ulong seed = 12;
const ulong epoch_size = 10000;

class SGDTestBase : public ::testing::Test {
 public:
  ArrayDouble w0;
  ArrayDouble noise;

  ModelPtr modelPtr;
};

class SGDTest : public SGDTestBase {
 protected:
  virtual void SetUp() {
    features = ArrayDouble2d(n_samples, n_features);
    labels = ArrayDouble(n_samples);

    w0 = ArrayDouble(n_features);
    noise = ArrayDouble(n_samples);

    std::mt19937 gen(42);
    std::uniform_real_distribution<> dis;

    for (ulong i = 0; i < w0.size(); ++i)
      w0[i] = dis(gen);

    for (ulong i = 0; i < noise.size(); ++i)
      noise[i] = dis(gen) * 0.00;

    for (ulong i = 0; i < features.size(); ++i)
      features[i] = dis(gen);

    for (ulong i = 0; i < labels.size(); ++i) {
      auto x = view_row(features, i);

      labels[i] = x.dot(w0) + noise[i];
    }

    modelPtr = std::make_shared<ModelLinReg>(features.as_sarray2d_ptr(), labels.as_sarray_ptr(), false);

    ArrayDouble starting_iterator(n_features);
    starting_iterator.init_to_zero();

    TICK_WARNING() << "w0 " << w0;
  }

  ArrayDouble2d features;
  ArrayDouble labels;
};

template <typename S>
void run(S& solver, SGDTestBase& fixture, const long unsigned int iterations = num_iterations) {
  solver.set_model(fixture.modelPtr);
  solver.set_rand_max(n_samples);

//  solver.set_prox(std::make_shared<ProxL2Sq>(1.0 / n_samples, false));

  ArrayDouble out(n_features);
  double diff = 1e10;
  for (ulong i = 0; i < num_iterations; ++i) {

    solver.solve();
    solver.get_iterate(out);

    diff = 0.0;
    for (ulong f_i = 0; f_i < n_features; ++f_i) {
      diff += std::abs(out[f_i] - fixture.w0[f_i]);
    }

  }

  TICK_WARNING() << "absdiff " << diff;
  TICK_WARNING() << "out " << out;
}

TEST_F(SGDTest, SGD) {
  SGD solver{
      epoch_size, 0.01, RandType::perm, initial_step_size, seed
  };

  run(solver, *this);
}

TEST_F(SGDTest, SGDAlt) {
  SGDAlt solver{
      epoch_size, 0.01, RandType::perm, initial_step_size, seed
  };

  run(solver, *this, num_iterations);
}

TEST_F(SGDTest, Minibatch) {
  SGDMinibatch solver{
      epoch_size, 0.01, RandType::perm, initial_step_size, seed
  };

  run(solver, *this);
}

TEST_F(SGDTest, Hogwild) {
  SGDHogwild solver{
      epoch_size, 0.01, RandType::perm, initial_step_size, seed
  };

  run(solver, *this);
}