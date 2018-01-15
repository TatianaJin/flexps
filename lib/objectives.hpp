#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "base/magic.hpp"
#include "lib/data_loader/data_store.hpp"
#include "lib/data_loader/labeled_sample.hpp"

namespace flexps {
namespace lib {

class Objective {  // TODO(tatiana) may wrap model and paramters
 public:
  using LabeledSample = TypedLabeledSample<float, float>;  // TODO(tatiana): use template?

  /**
   * @param num_params  The total number of parameters (including bias)
   */
  explicit Objective(int num_params) : num_params_(num_params) {}

  // SArray API
  virtual void get_gradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                            const third_party::SArray<float>& params, third_party::SArray<float>* delta,
                            int cardinality = 0) = 0;

  virtual float get_loss(DataStore<LabeledSample>& data_store, const third_party::SArray<float>& model) = 0;

  virtual void process_keys(third_party::SArray<Key>* keys) {
    keys->push_back(num_params_ - 1);  // add key for bias
  }

  virtual void all_keys(third_party::SArray<Key>* keys) {
    keys->resize(num_params_);
    std::iota(keys->begin(), keys->end(), 0);
  }

  virtual float predict(const LabeledSample& sample, const third_party::SArray<Key>& keys,
                        const third_party::SArray<float>& params) = 0;

  virtual float get_accuracy(DataStore<LabeledSample>& data_store, const third_party::SArray<float>& model) {
    return 0.0;
  }

  // std::vector API
  void get_gradient(const std::vector<LabeledSample*>& batch, const std::vector<Key>& keys,
                    const std::vector<float>& params, std::vector<float>* delta, int cardinality = 0) {
    third_party::SArray<float> delta_array;
    get_gradient(batch, third_party::SArray<Key>(keys), third_party::SArray<float>(params), &delta_array, cardinality);
    delta->assign(delta_array.begin(), delta_array.end());
  }

  float get_loss(DataStore<LabeledSample>& data_store, const std::vector<float>& model) {
    return get_loss(data_store, third_party::SArray<float>(model));
  }

  virtual void process_keys(std::vector<Key>* keys) {
    keys->push_back(num_params_ - 1);  // add key for bias
  }

  virtual void all_keys(std::vector<Key>* keys) {
    keys->resize(num_params_);
    std::iota(keys->begin(), keys->end(), 0);
  }

  float predict(const LabeledSample& sample, const std::vector<Key>& keys, const std::vector<float>& params) {
    return predict(sample, third_party::SArray<Key>(keys), third_party::SArray<float>(params));
  }

 protected:
  int num_params_ = 0;
};

class SigmoidObjective : public Objective {
 public:
  explicit SigmoidObjective(int num_params) : Objective(num_params){};

  float predict(const LabeledSample& sample, const third_party::SArray<Key>& keys,
                const third_party::SArray<float>& params) {
    auto& x = sample.x_;
    float pred_y = 0.0;

    if (params.size() == num_params_) {
      for (auto field : x) {
        pred_y += params[field.first] * field.second;
      }
    } else {
      CHECK_EQ(params.size(), keys.size());
      int i = 0;
      for (auto field : x) {
        while (keys[i] < field.first)
          i += 1;
        pred_y += params[i] * field.second;
      }
    }
    pred_y += params.back();  // intercept
    pred_y = 1. / (1. + exp(-1 * pred_y));

    return pred_y;
  }

  float get_accuracy(DataStore<LabeledSample>& data_store, const third_party::SArray<float>& model) {
    auto samples = data_store.Get();
    int accurate_count = 0;

    for (auto* sample : samples) {
      auto& x = sample->x_;
      float y = sample->y_;
      if (y < 0)
        y = 0.;
      float pred_y = 0.0f;
      for (auto& field : x) {
        pred_y += model[field.first] * field.second;
      }
      pred_y += model[num_params_ - 1];  // intercept
      pred_y = (pred_y > 0) ? 1 : 0;
      accurate_count += (int) y == (int) pred_y;
    }

    return (float) accurate_count / samples.size();
  }

  void get_gradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                    const third_party::SArray<float>& params, third_party::SArray<float>* delta, int cardinality = 0) {
    if (batch.empty())
      return;
    CHECK_EQ(delta->size(), keys.size());
    for (auto data : batch) {  // iterate over the data in the batch
      auto& x = data->x_;
      float y = data->y_;
      if (y < 0)
        y = 0.;

      float pred_y = predict(*data, keys, params);
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
      d /= static_cast<float>(cardinality);
    }
  }

  float get_loss(DataStore<LabeledSample>& data_store, const third_party::SArray<float>& model) {
    CHECK_EQ(num_params_, model.size());
    auto samples = data_store.Get();
    float count = samples.size();
    float loss = 0.0f;
    for (auto* sample : samples) {
      auto& x = sample->x_;
      float y = sample->y_;
      if (y < 0)
        y = 0.;
      float pred_y = 0.0f;
      for (auto& field : x) {
        pred_y += model[field.first] * field.second;
      }
      pred_y += model[num_params_ - 1];  // intercept
      pred_y = 1. / (1. + exp(-pred_y));
      if (y == 0) {
        loss += -log(1. - pred_y) / count;  // divide here to prevent overflow
      } else {                              // y == 1
        loss += -log(pred_y) / count;
      }
    }
    return loss;
  }
};

class LinearRegressionObjective : public Objective {
 public:
  explicit LinearRegressionObjective(int num_params) : Objective(num_params){};

  float predict(const LabeledSample& sample, const third_party::SArray<Key>& keys,
                const third_party::SArray<float>& params) {
    auto& x = sample.x_;
    float pred_y = 0.0;
    if (params.size() == num_params_) {
      for (auto field : x) {
        pred_y += params[field.first] * field.second;
      }
    } else {
      CHECK_EQ(params.size(), keys.size());
      int i = 0;
      for (auto field : x) {
        while (keys[i] < field.first)
          i += 1;
        pred_y += params[i] * field.second;
      }
    }

    pred_y += params.back();  // intercept
    return pred_y;
  }

  void get_gradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                    const third_party::SArray<float>& params, third_party::SArray<float>* delta, int cardinality = 0) {
    if (batch.empty())
      return;
    CHECK_EQ(delta->size(), keys.size());
    // 1. Calculate the sum of gradients
    for (auto data : batch) {  // iterate over the data in the batch
      auto& x = data->x_;
      float y = data->y_;
      float pred_y = predict(*data, keys, params);
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
      d /= static_cast<float>(cardinality);
    }
  }

  float get_loss(DataStore<LabeledSample>& data_store, const third_party::SArray<float>& model) {
    CHECK_EQ(num_params_, model.size());
    // 1. Calculate MSE on samples
    auto samples = data_store.Get();
    int count = 0;
    float loss = 0.0f;
    for (auto* sample : samples) {
      count += 1;

      auto& x = sample->x_;
      float y = sample->y_;
      float pred_y = 0.0f;
      for (auto& field : x) {
        pred_y += model[field.first] * field.second;
      }
      pred_y += model[num_params_ - 1];  // intercept
      float diff = pred_y - y;
      loss += diff * diff;
    }
    if (count != 0) {
      loss /= static_cast<float>(count);
    }

    return loss;
  }
};

class SVMObjective : public Objective {
 public:
  explicit SVMObjective(int num_params) : Objective(num_params){};
  SVMObjective(int num_params, float lambda) : Objective(num_params), lambda_(lambda){};

  float predict(const LabeledSample& sample, const third_party::SArray<Key>& keys,
                const third_party::SArray<float>& params) {
    CHECK_EQ(keys.size(), params.size());
    auto& x = sample.x_;
    float pred_y = 0.0;
    int i = 0;
    for (auto field : x) {
      while (keys[i] < field.first)
        i += 1;
      pred_y += params[i] * field.second;
    }
    pred_y += params.back();  // intercept

    return pred_y;
  }

  void get_gradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                    const third_party::SArray<float>& params, third_party::SArray<float>* delta, int cardinality = 0) {
    if (batch.empty())
      return;
    CHECK_EQ(delta->size(), keys.size());
    CHECK_EQ(params.size(), keys.size());
    // 1. Hinge loss gradients
    for (auto data : batch) {  // iterate over the data in the batch
      auto& x = data->x_;
      float y = data->y_;
      float pred_y = predict(*data, keys, params);
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
    // 2. ||w||^2 gradients TODO(tatiana): should be computed on servers
    for (int i = 0; i < keys.size() - 1; ++i) {  // omit bias
      (*delta)[i] /= static_cast<float>(batch_size);
      (*delta)[i] += lambda_ * params[i];
    }
  }

  float get_loss(DataStore<LabeledSample>& data_store, const third_party::SArray<float>& model) {
    CHECK_EQ(num_params_, model.size());
    // 1. Calculate hinge loss
    auto samples = data_store.Get();
    int count = 0;
    float loss = 0.0f;
    for (auto* sample : samples) {
      count += 1;

      auto& x = sample->x_;
      float y = sample->y_;
      float pred_y = 0.0f;
      for (auto& field : x) {
        pred_y += model[field.first] * field.second;
      }
      pred_y += model[num_params_ - 1];  // intercept
      loss += std::max(0., 1. - y * pred_y);
    }
    if (count != 0) {
      loss /= static_cast<float>(count);
    }

    // 2. Calculate ||w||^2
    float w_2 = 0.;
    for (float param : model) {
      w_2 += param * param;
    }
    loss += w_2 * lambda_;

    return loss;
  }

  inline void set_lambda(float lambda) { lambda_ = lambda; }

 private:
  float lambda_ = 0;  // hinge loss factor
};

}  // namespace lib
}  // namespace husky
