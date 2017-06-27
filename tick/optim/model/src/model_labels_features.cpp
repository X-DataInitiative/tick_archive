#include "model_labels_features.h"

// License: BSD 3 clause

ModelLabelsFeatures::ModelLabelsFeatures(SBaseArrayDouble2dPtr features,
                                         SArrayDoublePtr labels)
    : n_samples(labels.get() ? labels->size() : 0),
      n_features(features.get() ? features->n_cols() : 0),
      labels(labels),
      features(features),
      ready_columns_sparsity(false) {
  if (labels.get() && labels->size() != features->n_rows()) {
    std::stringstream ss;
    ss << "In ModelLabelsFeatures, number of labels is " << labels->size();
    ss << " while the features matrix has " << features->n_rows() << " rows.";
    throw std::invalid_argument(ss.str());
  }
}

void ModelLabelsFeatures::compute_columns_non_zeros(ArrayULong &out_columns_non_zeros) {
  if (features->is_sparse()) {
    if (out_columns_non_zeros.size() != n_features) {
      TICK_ERROR("given `out_columns_non_zeros` vector must match `n_features`")
    }
    out_columns_non_zeros.fill(0.);
    for (ulong i = 0; i < n_samples; ++i) {
      BaseArrayDouble features_i = get_features(i);
      for (ulong j = 0; j < features_i.size_sparse(); ++j) {
        // If the entry is indeed non-zero (nothing forbids to store zeros...) increment
        // the number of non-zeros of the columns
        if (features_i.data()[j] != 0) {
          out_columns_non_zeros[features_i.indices()[j]] += 1;
        }
      }
    }
  } else {
    TICK_ERROR("The features matrix is not sparse.")
  }
}
