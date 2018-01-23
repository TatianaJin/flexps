#pragma once

#include <iomanip>
#include <memory>

#include "base/third_party/sarray.h"
#include "driver/info.hpp"
#include "lib/data_loader/data_store.hpp"
#include "lib/objectives/objective.hpp"
#include "lib/optimizers/optimizer.hpp"
#include "lib/utils.hpp"

#include "lib/regularizer.hpp"

namespace flexps {
namespace lib {

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer(std::shared_ptr<Objective> objective, int report_interval) : Optimizer(objective, report_interval) {}

  void Train(const Info& info, int table_id, DataStore<LabeledSample>& data_store, const OptimizerConfig& config,
             int iter_offset = 0) override {
    // 1. Get KVClientTable for communication with server
    auto table = info.CreateKVClientTable<ValT>(table_id);

    // 2. Create BatchDataSampler for mini-batch SGD
    BatchIterator<LabeledSample> batch_data_sampler(data_store);
    batch_data_sampler.random_start_point();

    // 3. Main loop
    Timer train_timer(true);
    for (int iter = iter_offset; iter < config.num_iters + iter_offset; ++iter) {
      // a. Train
      ValT alpha = config.alpha / (iter / config.learning_rate_decay + 1);
      alpha = std::max(1e-5, alpha);

      Update(table, batch_data_sampler, alpha, config.batch_size);
      table.Clock();

      // b. Report loss on training samples
      if (report_interval_ != 0 && (iter + 1) % report_interval_ == 0) {
        train_timer.pause();
        if (info.worker_id == 0) {  // let the cluster leader do the report
          third_party::SArray<ValT> vals;
          third_party::SArray<Key> keys;
          objective_->AllKeys(&keys);
          // pull model
          table.Get(keys, &vals);
          // test with training samples
          auto loss = objective_->GetLoss(data_store.Get(), vals);
          LOG(INFO) << "Iter, Time, Loss: " << iter << "," << train_timer.elapsed_time() << "," << std::setprecision(15)
                    << loss;
        }
        table.Clock();
        train_timer.start();
      }
    }
    if (info.worker_id == 0) {  // let the cluster leader do the report
      third_party::SArray<ValT> vals;
      third_party::SArray<Key> keys;
      objective_->AllKeys(&keys);
      table.Get(keys, &vals);
      auto accuracy = objective_->GetAccuracy(data_store.Get(), vals);
      LOG(INFO) << "Accuracy: " << accuracy;
    }
    LOG(INFO) << "Total training time: " << train_timer.elapsed_time();
  }

 private:
  void Update(KVClientTable<ValT>& table, BatchIterator<LabeledSample>& batch_data_sampler, ValT alpha,
              int batch_size) {
    // 1. Prepare all the parameter keys in the batch
    auto keys_samples = batch_data_sampler.NextBatch(batch_size);
    auto& keys = keys_samples.first;
    objective_->ProcessKeys(&keys);
    third_party::SArray<ValT> params, delta;
    delta.resize(keys.size(), 0.0);

    // 2. Pull parameters
    table.Get(keys, &params);

    // 3. Calculate gradients
    objective_->GetGradient(keys_samples.second, keys, params, &delta);

    // 4. Adjust step size
    for (auto& d : delta) {
      d *= -alpha;
    }

    // 5. Push updates
    table.Add(keys, delta);
  }
};

}  // namespace lib
}  // namespace husky
