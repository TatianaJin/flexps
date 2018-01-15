#pragma once

#include <iomanip>
#include <memory>

#include "base/third_party/sarray.h"
#include "driver/info.hpp"
#include "lib/objectives.hpp"
#include "lib/regularizer.hpp"
#include "lib/utils.hpp"

namespace flexps {
namespace lib {

struct OptimizerConfig {
  int num_iters = 10;
  float alpha = 0.1;
  int batch_size = 10;
  int learning_rate_decay = 10;

  int regularizer = 0;
  float eta1 = 0.001;
  float eta2 = 0.001;
  // for svrg
  int num_epoches = 5;
  bool async = false;
  int gradient_table_id = -1;
  int cardinality = 0;
};

class Optimizer {
 public:
  using LabeledSample = Objective::LabeledSample;
  Optimizer(std::shared_ptr<Objective> objective, int report_interval)
      : objective_(objective), report_interval_(report_interval) {}

  virtual void Train(const Info& info, int table_id, DataStore<LabeledSample>& data_store,
                     const OptimizerConfig& config, int iter_offset = 0) = 0;

  std::shared_ptr<Objective> GetObjective() { return objective_; }

 protected:
  std::shared_ptr<Objective> objective_;
  int report_interval_ = 0;
};

/**
 * SVRG with sparse update (update reweighted using reverse dimension probability diag matrix)
 */
class SVRGOptimizer : public Optimizer {
 public:
  SVRGOptimizer(std::shared_ptr<Objective> objective, int report_interval) : Optimizer(objective, report_interval) {}

  void Train(const Info& info, int table_id, DataStore<LabeledSample>& data_store, const OptimizerConfig& config,
             int iter_offset = 0) override {
    // 1. Get KVClientTable for communication with server
    auto table = info.CreateKVClientTable<float>(table_id);
    auto gradient_table = info.CreateKVClientTable<float>(config.gradient_table_id);

    // 2. Create BatchDataSampler for mini-batch SGD
    BatchIterator<LabeledSample> batch_data_sampler(data_store);
    batch_data_sampler.random_start_point();

    // 3. Main loop
    Timer train_timer(true);

    third_party::SArray<float> dim_prob;
    for (int epoch = iter_offset; epoch < config.num_epoches + iter_offset; ++epoch) {  // outer iteration
      third_party::SArray<float> snapshot, delta_s;
      // A. Full gradient
      third_party::SArray<Key> keys;
      objective_->all_keys(&keys);
      auto samples = data_store.GetPtrs(info.local_id);
      delta_s.resize(keys.size(), 0.0);

      // Calculate dimension inverse probability diag matrix
      if (epoch == 0) {
        int num_params = keys.size();
        third_party::SArray<float> dim_count(num_params, 0.0);
        for (auto* sample : samples) {
          for (auto& idx_val : sample->x_) {
            dim_count[idx_val.first] += 1;
          }
        }
        gradient_table.Add(keys, dim_count);
        gradient_table.Clock();
        gradient_table.Get(keys, &dim_prob);
        for (int idx = 0; idx < num_params - 1; ++idx) {
          dim_prob[idx] = (float) config.cardinality / dim_prob[idx];
        }
      }

      if (config.async) {  // online svrg
        table.Get(keys, &snapshot);
        objective_->get_gradient(samples, keys, snapshot, &delta_s);
      } else {                       // sync full gradient step
        table.Get(keys, &snapshot);  // TODO(tatiana): shared pull
        objective_->get_gradient(samples, keys, snapshot, &delta_s, config.cardinality);

        gradient_table.Add(keys, delta_s);
        gradient_table.Clock();
        gradient_table.Get(keys, &delta_s);
      }

      LOG(INFO) << "[SVRGOptimizer] Epoch " << epoch << ": Full gradient step done";

      // B. inner iteration
      for (int iter = 0; iter < config.num_iters; ++iter) {
        // a. Train
        // FIXME(tatiana) float alpha = config.alpha / (iter / config.learning_rate_decay + 1);
        float alpha = config.alpha;

        update(table, batch_data_sampler, alpha, config, snapshot, delta_s, dim_prob, keys);
        table.Clock();

        if (info.worker_id == 0 && (iter + 1) % 10000 == 0) {
          LOG(INFO) << "SGD iteration " << (iter + 1);
        }
        // b. Report loss on training samples
        if (report_interval_ != 0 && (iter + 1) % report_interval_ == 0) {
          train_timer.pause();
          if (info.worker_id == 0) {  // let the cluster leader do the report
            third_party::SArray<float> vals;
            third_party::SArray<Key> keys;
            objective_->all_keys(&keys);
            // pull model
            table.Get(keys, &vals);
            // test with training samples
            auto loss = objective_->get_loss(data_store, vals);
            LOG(INFO) << "Iter, Time, Loss: " << iter << "," << train_timer.elapsed_time() << ","
                      << std::setprecision(15) << loss;
          }
          table.Clock();
          train_timer.start();
        }
      }
    }

    // Report accuracy and time
    if (info.local_id == 0) {
      third_party::SArray<float> vals;
      third_party::SArray<Key> keys;
      objective_->all_keys(&keys);
      table.Get(keys, &vals);
      auto accuracy = objective_->get_accuracy(data_store, vals);
      LOG(INFO) << "Worker " << info.worker_id << " Accuracy: " << accuracy;
      LOG(INFO) << "Worker " << info.worker_id << " Total training time: " << train_timer.elapsed_time();
    }
  }

 protected:
  void update(KVClientTable<float>& table, BatchIterator<LabeledSample>& batch_data_sampler, float alpha,
              const OptimizerConfig& config, const third_party::SArray<float>& snapshot,
              const third_party::SArray<float>& delta_s, const third_party::SArray<float>& dim_prob,
              const third_party::SArray<Key>& all_keys) {
    // 1. Prepare all the parameter keys in the batch
    auto keys_samples = batch_data_sampler.NextBatch(config.batch_size);
    auto& keys = keys_samples.first;
    objective_->process_keys(&keys);
    third_party::SArray<float> params, delta, gradient_s;
    delta.resize(keys.size(), 0.0);
    gradient_s.resize(keys.size(), 0.0);

    // 2. Pull parameters
    table.Get(keys, &params);

    // 3. Calculate gradients
    objective_->get_gradient(keys_samples.second, keys, params, &delta);
    objective_->get_gradient(keys_samples.second, keys, snapshot, &gradient_s);
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

class SGDOptimizer : public Optimizer {
 public:
  SGDOptimizer(std::shared_ptr<Objective> objective, int report_interval) : Optimizer(objective, report_interval) {}

  void Train(const Info& info, int table_id, DataStore<LabeledSample>& data_store, const OptimizerConfig& config,
             int iter_offset = 0) override {
    // 1. Get KVClientTable for communication with server
    auto table = info.CreateKVClientTable<float>(table_id);

    // 2. Create BatchDataSampler for mini-batch SGD
    BatchIterator<LabeledSample> batch_data_sampler(data_store);
    batch_data_sampler.random_start_point();

    // 3. Main loop
    Timer train_timer(true);
    for (int iter = iter_offset; iter < config.num_iters + iter_offset; ++iter) {
      // a. Train
      float alpha = config.alpha / (iter / config.learning_rate_decay + 1);
      alpha = std::max(1e-5f, alpha);
      update(table, batch_data_sampler, alpha, config.batch_size);
      table.Clock();

      // b. Report loss on training samples
      if (report_interval_ != 0 && (iter + 1) % report_interval_ == 0) {
        train_timer.pause();
        if (info.worker_id == 0) {  // let the cluster leader do the report
          third_party::SArray<float> vals;
          third_party::SArray<Key> keys;
          objective_->all_keys(&keys);
          // pull model
          table.Get(keys, &vals);
          // test with training samples
          auto loss = objective_->get_loss(data_store, vals);
          LOG(INFO) << "Iter, Time, Loss: " << iter << "," << train_timer.elapsed_time() << "," << std::setprecision(15)
                    << loss;
        }
        table.Clock();
        train_timer.start();
      }
    }
    if (info.worker_id == 0) {  // let the cluster leader do the report
      third_party::SArray<float> vals;
      third_party::SArray<Key> keys;
      objective_->all_keys(&keys);
      table.Get(keys, &vals);
      auto accuracy = objective_->get_accuracy(data_store, vals);
      LOG(INFO) << "Accuracy: " << accuracy;
    }
    LOG(INFO) << "Total training time: " << train_timer.elapsed_time();
  }

 private:
  void update(KVClientTable<float>& table, BatchIterator<LabeledSample>& batch_data_sampler, float alpha,
              int batch_size) {
    // 1. Prepare all the parameter keys in the batch
    auto keys_samples = batch_data_sampler.NextBatch(batch_size);
    auto& keys = keys_samples.first;
    objective_->process_keys(&keys);
    third_party::SArray<float> params, delta;
    delta.resize(keys.size(), 0.0);

    // 2. Pull parameters
    table.Get(keys, &params);

    // 3. Calculate gradients
    objective_->get_gradient(keys_samples.second, keys, params, &delta);

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
