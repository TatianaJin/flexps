#include "lib/objectives/linear_regression_objective.hpp"

namespace flexps {
namespace lib {

LinearRegressionObjective::LinearRegressionObjective(int num_dims) : Objective(num_dims) {}

ValT LinearRegressionObjective::Predict(const LabeledSample& sample, const third_party::SArray<ValT>& params,
                                        const third_party::SArray<Key>* keys) {
  auto& x = sample.x_;
  ValT pred_y = 0.0;

  if (keys == nullptr || params.size() == num_params_) {
    CHECK_EQ(params.size(), num_params_);
    for (auto field : x) {
      pred_y += params[field.first] * field.second;
    }
  } else {
    CHECK_EQ(params.size(), keys->size());
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

void LinearRegressionObjective::GetGradient(const std::vector<LabeledSample*>& batch,
                                            const third_party::SArray<Key>& keys,
                                            const third_party::SArray<ValT>& params, third_party::SArray<ValT>* delta,
                                            int cardinality) {
  if (batch.empty())
    return;
  CHECK_EQ(delta->size(), keys.size());
  // 1. Calculate the sum of gradients
  for (auto data : batch) {  // iterate over the data in the batch
    auto& x = data->x_;
    ValT y = data->y_;
    ValT pred_y = Predict(*data, params, &keys);
    int i = 0;
    for (auto field : x) {
      while (keys[i] < field.first)
        i += 1;
      (*delta)[i] += field.second * (pred_y - y);
    }
    (*delta)[delta->size() - 1] += pred_y - y;
  }

  // 2. Take average
  if (cardinality == 0)
    cardinality = batch.size();
  for (auto& d : *delta) {
    d /= static_cast<ValT>(cardinality);
  }
}

ValT LinearRegressionObjective::GetLoss(const std::vector<LabeledSample*>& samples,
                                        const third_party::SArray<ValT>& model, ValT denominator,
                                        const third_party::SArray<Key>* keys) {
  if (denominator == 0)
    denominator = samples.size();
  ValT loss = 0.0f;

  for (auto* sample : samples) {
    ValT y = (sample->y_ < 0) ? -1. : 1.;
    ValT diff = Predict(*sample, model, keys) - y;
    loss += diff * diff / denominator;
  }

  return loss;
}

// the same as loss
ValT LinearRegressionObjective::GetAccuracy(const std::vector<LabeledSample*>& samples,
                                            const third_party::SArray<ValT>& model,
                                            const third_party::SArray<Key>* keys, ValT denominator) {
  return GetLoss(samples, model, denominator, keys);
}

}  // namespace lib
}  // namespace husky
