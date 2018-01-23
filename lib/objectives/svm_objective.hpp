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

class SVMObjective : public Objective {
 public:
  explicit SVMObjective(int num_dims) : Objective(num_dims){};
  SVMObjective(int num_dims, ValT lambda) : Objective(num_dims), lambda_(lambda){};

  ValT Predict(const LabeledSample& sample, const third_party::SArray<ValT>& params,
               const third_party::SArray<Key>* keys = nullptr) override {
    ValT pred_y = 0.0;
    auto& x = sample.x_;

    if (keys == nullptr || params.size() == num_params_) {
      CHECK_EQ(params.size(), num_params_);
      for (auto field : x) {
        pred_y += params[field.first] * field.second;
      }
    } else {
      CHECK_EQ(keys->size(), params.size());
      int i = 0;
      for (auto field : x) {
        while ((*keys)[i] < field.first)
          i += 1;
        pred_y += params[i] * field.second;
      }
    }

    pred_y += params.back();  // intercept
    return pred_y;
  }

  void GetGradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                   const third_party::SArray<ValT>& params, third_party::SArray<ValT>* delta,
                   int cardinality = 0) override {
    if (batch.empty())
      return;
    CHECK_EQ(delta->size(), keys.size());
    CHECK_EQ(params.size(), keys.size());
    // 1. Hinge loss gradients
    for (auto data : batch) {  // iterate over the data in the batch
      auto& x = data->x_;
      ValT y = data->y_;
      ValT pred_y = Predict(*data, params, &keys);
      if (y * pred_y < 1) {  // in soft margin
        int i = 0;
        for (auto field : x) {
          while (keys[i] < field.first)
            i += 1;
          (*delta)[i] -= field.second * y;
        }
        (*delta)[delta->size() - 1] -= y;
      }
    }

    int batch_size = batch.size();
    // 2. ||w||^2 gradients FIXME(tatiana): should be computed on servers or use probability matrix
    for (int i = 0; i < keys.size() - 1; ++i) {  // omit bias
      (*delta)[i] /= static_cast<ValT>(batch_size);
      (*delta)[i] += lambda_ * params[i];
    }
  }

  /**
   * Only the slack penalty, not including the margin
   */
  ValT GetLoss(const std::vector<LabeledSample*>& samples, const third_party::SArray<ValT>& model, ValT denominator = 0,
               const third_party::SArray<Key>* keys = nullptr) override {
    if (samples.empty())
      return 0.;

    ValT loss = 0.0f;
    if (denominator == 0)
      denominator = samples.size();

    for (auto* sample : samples) {
      ValT y = sample->y_;
      if (y == 0)
        y = -1;
      loss += std::max(0., 1. - y * Predict(*sample, model, keys)) / denominator;
    }

    return loss;
  }

  inline void set_lambda(ValT lambda) { lambda_ = lambda; }

 private:
  ValT lambda_ = 0;  // hinge loss factor
};

}  // namespace lib
}  // namespace husky
