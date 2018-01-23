#pragma once

#include <set>
#include <vector>

#include "base/magic.hpp"
#include "base/third_party/sarray.h"

namespace flexps {
namespace lib {

template <typename Sample>
class DataStore {
 public:
  DataStore(int n_slots) { samples_.resize(n_slots); }

  void Push(int tid, Sample&& sample) { samples_[tid].push_back(std::move(sample)); }

  const std::vector<Sample>& Get(int slot_id) const {
    CHECK_LT(slot_id, samples_.size());
    return samples_[slot_id];
  }

  std::vector<Sample*> GetPtrs(int slot_id) {
    CHECK_LT(slot_id, samples_.size());
    std::vector<Sample*> ret;
    ret.reserve(samples_[slot_id].size());
    for (auto& sample : samples_[slot_id]) {
        ret.push_back(&sample);
    }
    return ret;
  }

  std::vector<Sample*> Get() {
    std::vector<Sample*> ret;
    for (auto& slot : samples_) {
      for (auto& sample : slot) {
        ret.push_back(&sample);
      }
    }
    return ret;
  }

 protected:
  std::vector<std::vector<Sample>> samples_;
};

template <typename Sample>
class BatchIterator {
 public:
  BatchIterator(DataStore<Sample>& data_store) : samples_(data_store.Get()) {}

  void random_start_point() { sample_idx_ = rand() % samples_.size(); }

  const std::vector<Sample*>& GetSamples() const { return samples_; }

  std::pair<third_party::SArray<Key>, std::vector<Sample*>> NextBatch(int batch_size) {
    std::pair<third_party::SArray<Key>, std::vector<Sample*>> ret;
    ret.second.reserve(batch_size);
    std::set<Key> keys;

    for (int i = 0; i < batch_size; ++i) {
      ret.second.push_back(samples_[sample_idx_]);
      for (auto& field : samples_[sample_idx_]->x_) {
        keys.insert(field.first);
      }
      ++sample_idx_;

      sample_idx_ %= samples_.size();
    }

    ret.first = std::vector<Key>{keys.begin(), keys.end()};
    return ret;
  }

 protected:
  int sample_idx_ = 0;
  std::vector<Sample*> samples_;
};

}  // namespace lib
}  // namespace flexps
