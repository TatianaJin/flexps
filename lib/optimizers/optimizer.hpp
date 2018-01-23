#pragma once

#include <iomanip>
#include <memory>

#include "base/magic.hpp"
#include "base/third_party/sarray.h"
#include "driver/info.hpp"
#include "lib/data_loader/data_store.hpp"
#include "lib/objectives/objective.hpp"
#include "lib/regularizer.hpp"
#include "lib/utils.hpp"

namespace flexps {
namespace lib {

struct OptimizerConfig {
  int num_iters = 10;
  ValT alpha = 0.1;
  int batch_size = 10;
  int learning_rate_decay = 10;

  int regularizer = 0;
  ValT eta1 = 0.001;
  ValT eta2 = 0.001;

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

}  // namespace lib
}  // namespace husky
