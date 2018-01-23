#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "base/magic.hpp"
#include "lib/data_loader/typed_labeled_sample.hpp"
#include "lib/objectives/objective.hpp"

namespace flexps {
namespace lib {

class LinearRegressionObjective : public Objective {
 public:
  explicit LinearRegressionObjective(int num_dims);

  ValT Predict(const LabeledSample& sample, const third_party::SArray<ValT>& params,
               const third_party::SArray<Key>* keys = nullptr) override;

  void GetGradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                   const third_party::SArray<ValT>& params, third_party::SArray<ValT>* delta,
                   int cardinality = 0) override;

  ValT GetLoss(const std::vector<LabeledSample*>& samples, const third_party::SArray<ValT>& model, ValT denominator = 0,
               const third_party::SArray<Key>* keys = nullptr) override;

  ValT GetAccuracy(const std::vector<LabeledSample*>& samples, const third_party::SArray<ValT>& model,
                   const third_party::SArray<Key>* keys = nullptr, ValT denominator = 0) override;
};

}  // namespace lib
}  // namespace husky
