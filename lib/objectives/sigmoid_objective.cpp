#include "lib/objectives/sigmoid_objective.hpp"

#include <cmath>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "base/magic.hpp"
#include "base/third_party/sarray.h"

namespace flexps {
namespace lib {

SigmoidObjective::SigmoidObjective(int num_dims) : Objective(num_dims) {}

ValT SigmoidObjective::Predict(const LabeledSample& sample, const third_party::SArray<ValT>& params,
                               const third_party::SArray<Key>* keys) {
  return Predict(sample, params, true, keys);
}

ValT SigmoidObjective::Predict(const LabeledSample& sample, const third_party::SArray<ValT>& params, bool sigmoid,
                               const third_party::SArray<Key>* keys) {
  auto& x = sample.x_;
  ValT pred_y = 0.0;

  if (keys == nullptr || params.size() == num_params_) {
    CHECK_EQ(params.size(), num_params_)
        << "[SigmoidObjective] Predict: the given model is not complete but no keys are given";
    for (auto field : x) {
      pred_y += params[field.first] * field.second;
    }
  } else {
    CHECK_EQ(params.size(), keys->size()) << "[SigmoidObjective] Predict: keys size and model size are different";
    int i = 0;
    for (auto field : x) {
      while ((*keys)[i] < field.first)
        i += 1;
      pred_y += params[i] * field.second;
    }
  }

  pred_y += params.back();  // intercept

  if (sigmoid) {
    pred_y = 1. / (1. + exp(-1 * pred_y));
  }

  return pred_y;
}

ValT SigmoidObjective::GetAccuracy(const std::vector<LabeledSample*>& samples, const third_party::SArray<ValT>& model,
                                   const third_party::SArray<Key>* keys, ValT denominator) {
  if (samples.empty())
    return 0.f;
  ValT accurate_count = 0;

  if (denominator == 0)
    denominator = samples.size();

  for (auto* sample : samples) {
    auto pred_y = Predict(*sample, model, false, keys);
    pred_y = (pred_y > 0) ? 1 : 0;
    ValT y = sample->y_;
    if (y < 0)
      y = 0.;
    accurate_count += (int) y == (int) pred_y;
  }

  return accurate_count / denominator;
}

void SigmoidObjective::GetGradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                                   const third_party::SArray<ValT>& params, third_party::SArray<ValT>* delta,
                                   int cardinality) {
  if (batch.empty())
    return;
  CHECK_EQ(delta->size(), keys.size());
  for (auto data : batch) {  // iterate over the data in the batch
    auto& x = data->x_;
    ValT y = data->y_;
    if (y < 0)
      y = 0.;

    ValT pred_y = Predict(*data, params, &keys);
    int i = 0;
    for (auto field : x) {
      while (keys[i] < field.first)
        i += 1;
      (*delta)[i] += field.second * (pred_y - y);
    }
    (*delta)[keys.size() - 1] += pred_y - y;
  }

  // Take average
  if (cardinality == 0)
    cardinality = batch.size();
  for (auto& d : *delta) {
    d /= static_cast<ValT>(cardinality);
  }
}

ValT SigmoidObjective::GetLoss(const std::vector<LabeledSample*>& samples, const third_party::SArray<ValT>& model,
                               ValT denominator, const third_party::SArray<Key>* keys) {
  if (samples.empty())
    return 0.f;
  if (denominator == 0)
    denominator = samples.size();
  ValT loss = 0.0f;

  for (auto* sample : samples) {
    ValT y = (sample->y_ < 0) ? -1. : 1.;
    loss += log(1 + exp(-y * Predict(*sample, model, false, keys))) / denominator;
  }

  return loss;
}

}  // namespace lib
}  // namespace husky
