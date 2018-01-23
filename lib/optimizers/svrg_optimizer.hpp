#pragma once

#include <iomanip>
#include <memory>

#include "base/third_party/sarray.h"
#include "driver/info.hpp"
#include "lib/data_loader/data_store.hpp"
#include "lib/objectives/objective.hpp"
#include "lib/optimizers/optimizer.hpp"
#include "lib/regularizer.hpp"
#include "lib/utils.hpp"

namespace flexps {
namespace lib {

/**
 * SVRG with sparse update (update reweighted using reverse dimension probability diag matrix)
 */
class SVRGOptimizer : public Optimizer {
 public:
  SVRGOptimizer(std::shared_ptr<Objective> objective, int report_interval) : Optimizer(objective, report_interval) {}

  void GetDimensionOccurrence(third_party::SArray<ValT>* dim_prob, KVClientTable<ValT>& table,
                              const std::vector<LabeledSample*>& samples) {
    for (auto* sample : samples) {
      for (auto& idx_val : sample->x_) {
        (*dim_prob)[idx_val.first] += 1;
      }
    }

    third_party::SArray<Key> keys(dim_prob->size());
    std::iota(keys.begin(), keys.end(), 0);

    table.Add(keys, *dim_prob);
    table.Clock();
    table.Get(keys, dim_prob);
  }

  void Train(const Info& info, int table_id, DataStore<LabeledSample>& data_store, const OptimizerConfig& config,
             int iter_offset = 0) override {
    // 1. Get KVClientTable for communication with server
    auto table = info.CreateKVClientTable<ValT>(table_id);
    auto gradient_table = info.CreateKVClientTable<ValT>(config.gradient_table_id);

    // 2. Create BatchDataSampler for mini-batch SGD
    BatchIterator<LabeledSample> batch_data_sampler(data_store);
    batch_data_sampler.random_start_point();

    // 3. Main loop
    Timer train_timer(true);

    auto samples = data_store.GetPtrs(info.local_id);

    // Calculate dimension inverse probability diag matrix
    third_party::SArray<ValT> dim_prob(objective_->GetNumFeatures(), 0);
    GetDimensionOccurrence(&dim_prob, gradient_table, samples);
    for (int idx = 0; idx < objective_->GetNumFeatures(); ++idx) {
      dim_prob[idx] = (ValT) config.cardinality / dim_prob[idx];
    }

    for (int epoch = iter_offset; epoch < config.num_epoches + iter_offset; ++epoch) {  // outer iteration
      third_party::SArray<ValT> snapshot, delta_s;
      // A. Full gradient
      third_party::SArray<Key> keys;
      objective_->AllKeys(&keys);
      delta_s.resize(keys.size(), 0.0);

      if (config.async) {  // online svrg
        table.Get(keys, &snapshot);
        objective_->GetGradient(samples, keys, snapshot, &delta_s);
      } else {                       // sync full gradient step
        table.Get(keys, &snapshot);  // TODO(tatiana): shared pull
        objective_->GetGradient(samples, keys, snapshot, &delta_s, config.cardinality);

        gradient_table.Add(keys, delta_s);
        gradient_table.Clock();
        gradient_table.Get(keys, &delta_s);
      }

      LOG(INFO) << "[SVRGOptimizer] Epoch " << epoch << ": Full gradient step done";

      // B. inner iteration
      for (int iter = 0; iter < config.num_iters; ++iter) {
        Update(table, batch_data_sampler, config.alpha, config, snapshot, delta_s, dim_prob, keys);
        table.Clock();
      }
    }

    // Report time
    if (info.local_id == 0) {
      LOG(INFO) << "Worker " << info.worker_id << " Total training time: " << train_timer.elapsed_time();
    }
  }

 protected:
  void Update(KVClientTable<ValT>& table, BatchIterator<LabeledSample>& batch_data_sampler, ValT alpha,
              const OptimizerConfig& config, const third_party::SArray<ValT>& snapshot,
              const third_party::SArray<ValT>& delta_s, const third_party::SArray<ValT>& dim_prob,
              const third_party::SArray<Key>& all_keys) {
    // 1. Prepare all the parameter keys in the batch
    auto keys_samples = batch_data_sampler.NextBatch(config.batch_size);
    auto& keys = keys_samples.first;
    objective_->ProcessKeys(&keys);
    third_party::SArray<ValT> params, delta, gradient_s;
    delta.resize(keys.size(), 0.0);
    gradient_s.resize(keys.size(), 0.0);

    // 2. Pull parameters
    table.Get(keys, &params);

    // 3. Calculate gradients
    objective_->GetGradient(keys_samples.second, keys, params, &delta);
    objective_->GetGradient(keys_samples.second, keys, snapshot, &gradient_s);
    for (int i = 0; i < keys.size(); ++i) {
      delta[i] += -gradient_s[i] + delta_s[keys[i]] * dim_prob[keys[i]];
      auto temp_x = params[i] - delta[i] * alpha;
      delta[i] = proximal_operator(config.regularizer, temp_x, dim_prob[keys[i]] * alpha, config.eta1, config.eta2) -
                 params[i];
    }

    // 4. Push updates
    table.Add(keys, delta);
  }
};

}  // namespace lib
}  // namespace husky
