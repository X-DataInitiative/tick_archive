#include <random>

#include <gtest/gtest.h>

#include <tick/optim/model/src/linreg.h>
#include <tick/optim/solver/src/sgd_minibatch.h>
#include <prox_l2sq.h>

TEST(SGD, Dims) {
  const ulong n_features = 5;
  const ulong n_samples = 10000;
  const double initial_step_size = 50.0;

  const ulong seed = 12;
  const ulong epoch_size = n_samples;

  SGDMinibatch solver{
      epoch_size, 0.01, RandType::perm, initial_step_size, seed
  };

  ArrayDouble2d features(n_samples, n_features);
  ArrayDouble labels(n_samples);

  ArrayDouble w0(n_features);
  ArrayDouble noise(n_samples);

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

  ModelPtr modelPtr = std::make_shared<ModelLinReg>(features.as_sarray2d_ptr(), labels.as_sarray_ptr(), false);

  ArrayDouble starting_iterator(n_features);
  starting_iterator.init_to_zero();

  solver.set_model(modelPtr);

  solver.set_rand_max(n_samples);

//  ProxPtr proxPtr = std::make_shared<ProxL2Sq>(0.0, false);
//  solver.set_prox(proxPtr);

//  solver.set_starting_iterate(starting_iterator);

  TICK_WARNING() << "step " << initial_step_size;
  TICK_WARNING() << "w0 " << w0;

  for (ulong i = 0; i < 100; ++i) {
    solver.solve();
  }

  ArrayDouble out(n_features);
  solver.get_iterate(out);

  TICK_WARNING() << "out " << out;
}