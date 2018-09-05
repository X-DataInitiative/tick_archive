// License: BSD 3 clause

#ifndef LIB_INCLUDE_TICK_ONLINE_ONLINE_FOREST_CLASSIFIER_H_
#define LIB_INCLUDE_TICK_ONLINE_ONLINE_FOREST_CLASSIFIER_H_

#include <cmath>
#include <iomanip>
#include "tick/base/base.h"
#include "tick/random/rand.h"

// TODO: change the Dirichlet parameter
// TODO: feature_importances with a nullptr by default
// TODO: subsample parameter, default 0.5

// TODO: tree aggregation
// TODO: memory optimization (a FeatureSplitter), maximum (sizeof(uint8_t)
// splits)), a set of current splits
// TODO: only binary features version ?

// TODO: add a min_sample_split option
// TODO: column subsampling and feature selection

// TODO: class balancing: sample_weights or class_weight option

enum class CriterionClassifier {
  log = 0,
};

enum class FeatureImportanceType { no = 0, estimated = 1, given = 2 };

class TreeClassifier;

/*********************************************************************************
 * NodeClassifier
 *********************************************************************************/

class NodeClassifier {
 protected:
  // Tree containing the node
  TreeClassifier &_tree;
  // true if the node is a leaf
  bool _is_leaf = true;
  // depth of the node in the tree
  uint8_t _depth = 0;
  // Number of samples in the node
  uint32_t _n_samples = 0;
  // Index of the parent
  uint32_t _parent = 0;
  // Index of the left child
  uint32_t _left = 0;
  // Index of the right child
  uint32_t _right = 0;
  // Index of the feature used for the split
  uint32_t _feature = 0;
  // The label of the sample saved in the node
  float _y_t = 0;
  // Logarithm of the aggregation weight for the node
  float _weight = 0;
  // Logarithm of the aggregation weight for the sub-tree starting at this node
  float _weight_tree = 0;
  // Threshold used for the split
  float _threshold = 0;
  // Time of creation of the node
  float _time = 0;
  // Counts the number of sample seen in each class
  ArrayUInt _counts;
  // Range of the features
  ArrayFloat _features_min, _features_max;

  // Update range of the seen features
  void update_range(const ArrayDouble &x_t);
  // Update n_samples
  void update_n_samples();
  // Update to apply to a node when going forward in the tree (towards leaves)
  float update_weight(const ArrayDouble &x_t, const double y_t);
  // Update the prediction of the label
  void update_count(const double y_t);

 public:
  NodeClassifier(TreeClassifier &tree, uint32_t parent, float time = 0);
  NodeClassifier(const NodeClassifier &node);
  NodeClassifier(const NodeClassifier &&node);
  NodeClassifier &operator=(const NodeClassifier &);
  NodeClassifier &operator=(const NodeClassifier &&) = delete;

  // Computation of log( (e^a + e^b) / 2) in an overproof way
  inline static float log_sum_2_exp(const float a, const float b) {
    // TODO: if |a - b| > 50 skip
    if (a > b) {
      return a + std::log((1 + std::exp(b - a)) / 2);
    }
    return b + std::log((1 + std::exp(a - b)) / 2);
  }

  // Update the prediction of the label
  void update_downwards(const ArrayDouble &x_t, const double y_t, bool do_update_weight);
  // Update to apply to a node when going upward in the tree (towards the root)
  void update_weight_tree();

  bool is_dirac(const double y_t);

  // Get the index of the child node containing x_t
  uint32_t get_child(const ArrayDouble &x_t);

  // Predict function (average of the labels of samples that passed through the node)
  void predict(ArrayDouble &scores) const;
  // Loss function used for aggregation
  float loss(const double y_t);
  // Score of the node when the true label is y
  float score(uint8_t y) const;
  // Compute the exntension of the range of the node when adding x_t.
  // Output is in intensities, and returns the sum of the extensions
  void compute_range_extension(const ArrayDouble &x_t, ArrayFloat &extensions,
                               float &extensions_sum, float &extensions_max);

  // Get node at index in the tree
  inline NodeClassifier &node(uint32_t index) const;

  // Get number of features
  inline uint32_t n_features() const;
  // Number of classes
  inline uint8_t n_classes() const;
  // Step to use for aggregation
  inline float step() const;
  // Dirichlet prior
  inline float dirichlet() const;
  // Print the node
  void print();

  inline void increment_depth();

  inline uint32_t parent() const;
  inline NodeClassifier &parent(uint32_t parent);
  inline uint32_t left() const;
  inline NodeClassifier &left(uint32_t left);
  inline uint32_t right() const;
  inline NodeClassifier &right(uint32_t right);
  inline bool is_leaf() const;
  inline NodeClassifier &is_leaf(bool is_leaf);
  inline uint32_t feature() const;
  inline NodeClassifier &feature(uint32_t feature);
  inline float threshold() const;
  inline NodeClassifier &threshold(float threshold);
  inline float time() const;
  inline NodeClassifier &time(float time);
  inline uint8_t depth() const;
  inline NodeClassifier &depth(uint8_t depth);
  inline float features_min(const uint32_t j) const;
  inline float features_max(const uint32_t j) const;
  inline uint32_t n_samples() const;
  inline bool use_aggregation() const;
  inline float weight() const;
  inline float weight_tree() const;
};

class OnlineForestClassifier;

/*********************************************************************************
 * TreeClassifier
 *********************************************************************************/

class TreeClassifier {
 protected:
  // The forest of the tree
  OnlineForestClassifier &forest;
  // Number of features
  uint32_t _n_features;
  // Number of classes
  uint8_t _n_classes;
  // Number of nodes in the tree
  uint32_t _n_nodes = 0;
  // Iteration counter
  uint32_t iteration = 0;
  // Nodes of the tree
  std::vector<NodeClassifier> nodes = std::vector<NodeClassifier>();

  // Create the root node
  void create_root();
  // Add nodes in the tree
  uint32_t add_node(uint32_t parent, float time = 0);

  // A vector containing estimated feature importances
  ArrayFloat feature_importances_;
  // A vector used for computations
  ArrayFloat intensities;

  void split_node(uint32_t node_index, const float split_time, const float threshold,
                  const uint32_t feature, const bool is_right_extension);

  // Get the index of the leaf containing x_t
  uint32_t get_leaf(const ArrayDouble &x_t);

  uint32_t go_downwards(const ArrayDouble &x_t, double y_t);

  float compute_split_time(uint32_t node_index, const ArrayDouble &x_t, const double y_t);

  void go_upwards(uint32_t leaf_index);

  void split_node(uint32_t node_index, const ArrayDouble &x_t, const ArrayFloat &intensities);

  void update_depth(uint32_t node_index, uint8_t depth);

 public:
  explicit TreeClassifier(OnlineForestClassifier &forest);
  TreeClassifier(const TreeClassifier &tree);
  TreeClassifier(const TreeClassifier &&tree);
  TreeClassifier &operator=(const TreeClassifier &) = delete;
  TreeClassifier &operator=(const TreeClassifier &&) = delete;

  // Reserve nodes in the tree in advance
  void reserve_nodes(uint32_t n_nodes);

  void fit(const ArrayDouble &x_t, double y_t);
  void predict(const ArrayDouble &x_t, ArrayDouble &scores, bool use_aggregation);

  // Give the depth of the path for x_t
  uint32_t get_path_depth(const ArrayDouble &x_t);
  // Get the path of x_t
  void get_path(const ArrayDouble &x_t, SArrayUIntPtr path);

  void get_aggregate_path(const SArrayDoublePtr features, SArrayDouble2dPtr node_scores,
                          SArrayDoublePtr aggregation_weights);

  inline uint32_t n_features() const;
  inline uint8_t n_classes() const;
  inline uint32_t n_nodes() const;
  inline uint32_t n_nodes_reserved() const;
  uint32_t n_leaves() const;
  inline float step() const;
  inline float dirichlet() const;

  void print();

  inline CriterionClassifier criterion() const;
  inline bool use_aggregation() const;
  FeatureImportanceType feature_importance_type() const;
  float feature_importance(const uint32_t j) const;
  float given_feature_importance(const uint32_t j) const;

  NodeClassifier &node(uint32_t index) { return nodes[index]; }

  inline ArrayFloat &feature_importances() { return feature_importances_; }

  static void show_vector(const ArrayDouble x, int precision = 2) {
    std::cout << "[";
    for (ulong j = 0; j < x.size(); ++j) {
      std::cout << " " << std::setprecision(precision) << x[j];
    }
    std::cout << " ]" << std::endl;
  }

  void get_flat_nodes(SArrayUIntPtr nodes_parent, SArrayUIntPtr nodes_left,
                      SArrayUIntPtr nodes_right, SArrayUIntPtr nodes_feature,
                      SArrayFloatPtr nodes_threshold, SArrayFloatPtr nodes_time,
                      SArrayUShortPtr nodes_depth, SArrayFloat2dPtr nodes_features_min,
                      SArrayFloat2dPtr nodes_features_max, SArrayUIntPtr nodes_n_samples,
                      SArrayFloatPtr nodes_weight, SArrayFloatPtr nodes_weight_tree,
                      SArrayUShortPtr nodes_is_leaf, SArrayUInt2dPtr nodes_counts);
};

/*********************************************************************************
 * OnlineForestClassifier
 *********************************************************************************/

class OnlineForestClassifier {
 private:
  // Number of features
  uint32_t _n_features;
  // Number of classes in the classification problem
  uint8_t _n_classes;
  // Number of Trees in the forest
  uint8_t _n_trees;
  // Step-size used for aggregation
  float _step;

  bool _estimate_feature_importances;

  // A vector of given feature importances (not estimated)
  ArrayFloat _given_feature_importances;

  // CriterionClassifier used for splitting (not used for now)
  CriterionClassifier _criterion;
  //
  FeatureImportanceType _feature_importance_type;
  //
  bool _use_aggregation;
  //
  float _dirichlet;

  bool _split_pure;
  int32_t _max_nodes;
  float _min_extension_size;

  int32_t _min_samples_split, _max_features;

  // Number of threads to use for parallel growing of trees
  int32_t _n_threads;
  // Seed for random number generation
  int _seed;
  // Verbose things or not
  bool _verbose;
  // Iteration counter
  uint32_t _iteration;
  // The list of trees in the forest
  std::vector<TreeClassifier> trees;
  // Random number generator for feature and threshold sampling
  Rand rand;

  // Create trees
  void create_trees();

  void check_n_features(uint32_t n_features, bool predict) const;
  inline void check_label(double label) const;

 public:
  OnlineForestClassifier(uint32_t n_features, uint8_t n_classes, uint8_t n_trees, float step,
                         CriterionClassifier criterion,
                         FeatureImportanceType feature_importance_type, bool use_aggregation,
                         float dirichlet, bool split_pure, int32_t max_nodes,
                         float min_extension_size, int32_t min_samples_split, int32_t max_features,
                         int32_t n_threads, int seed, bool verbose);
  virtual ~OnlineForestClassifier();

  void fit(const SArrayDouble2dPtr features, const SArrayDoublePtr labels);
  void predict(const SArrayDouble2dPtr features, SArrayDouble2dPtr scores);

  inline uint32_t sample_feature(const ArrayFloat &prob);
  inline float sample_exponential(float intensity);
  inline float sample_threshold(float left, float right);

  void clear();
  void print();

  uint32_t n_samples() const;
  uint32_t n_features() const;
  uint8_t n_classes() const;

  uint8_t n_trees() const;
  bool use_aggregation() const;
  float step() const;
  OnlineForestClassifier &step(const float step);
  bool split_pure() const;
  int32_t max_nodes() const;
  float min_extension_size() const;
  int32_t min_samples_split() const;
  int32_t max_features() const;

  float dirichlet() const;
  OnlineForestClassifier &dirichlet(const float dirichlet);
  bool verbose() const;
  OnlineForestClassifier &verbose(bool verbose);
  CriterionClassifier criterion() const;
  OnlineForestClassifier &criterion(CriterionClassifier criterion);
  FeatureImportanceType feature_importance_type() const;

  double given_feature_importances(const ulong j) const;

  int32_t n_threads() const;
  OnlineForestClassifier &n_threads(int32_t n_threads);
  int seed() const;
  OnlineForestClassifier &seed(int seed);

  void n_nodes(SArrayUIntPtr n_nodes_per_tree);
  void n_nodes_reserved(SArrayUIntPtr n_reserved_nodes_per_tree);
  void n_leaves(SArrayUIntPtr n_leaves_per_tree);

  OnlineForestClassifier &given_feature_importances(const ArrayDouble &feature_importances);

  void get_feature_importances(SArrayDoublePtr feature_importances);

  // Get the path for of a tree for a single vector of features
  // void get_path(uint8_t tree, const SArrayDoublePtr features);

  // Give the depth of the path for x_t
  uint32_t get_path_depth(const uint8_t tree, const SArrayDoublePtr x_t);

  // Get the path of x_t
  void get_path(const uint8_t tree, const SArrayDoublePtr x_t, SArrayUIntPtr path);

  void get_flat_nodes(uint8_t tree, SArrayUIntPtr nodes_parent, SArrayUIntPtr nodes_left,
                      SArrayUIntPtr nodes_right, SArrayUIntPtr nodes_feature,
                      SArrayFloatPtr nodes_threshold, SArrayFloatPtr nodes_time,
                      SArrayUShortPtr nodes_depth, SArrayFloat2dPtr nodes_features_min,
                      SArrayFloat2dPtr nodes_features_max, SArrayUIntPtr nodes_n_samples,
                      SArrayFloatPtr nodes_weight, SArrayFloatPtr nodes_weight_tree,
                      SArrayUShortPtr nodes_is_leaf, SArrayUInt2dPtr nodes_counts);
};

#endif  // LIB_INCLUDE_TICK_ONLINE_ONLINE_FOREST_CLASSIFIER_H_
