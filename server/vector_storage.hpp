#pragma once

#include "base/message.hpp"
#include "base/third_party/range.h"
#include "server/abstract_storage.hpp"

#include "glog/logging.h"

#include <fstream>
#include <vector>

namespace flexps {

template <typename Val>
class VectorStorage : public AbstractStorage {
 public:
  /*
   * The storage is in charge of range [range.begin(), range.end()).
   */
  VectorStorage(third_party::Range range, uint32_t chunk_size = 1)
      : range_(range), storage_(range.size(), Val()), chunk_size_(chunk_size) {
    CHECK_LE(range_.begin(), range_.end());
  }

  virtual void SubAdd(const third_party::SArray<Key>& typed_keys, const third_party::SArray<char>& vals) override {
    auto typed_vals = third_party::SArray<Val>(vals);
    for (size_t index = 0; index < typed_keys.size(); index++) {
      CHECK_GE(typed_keys[index], range_.begin());
      CHECK_LT(typed_keys[index], range_.end());
      storage_[typed_keys[index] - range_.begin()] += typed_vals[index];
    }
  }

  virtual void SubAddChunk(const third_party::SArray<Key>& typed_keys, const third_party::SArray<char>& vals) override {
    auto typed_vals = third_party::SArray<Val>(vals);
    CHECK_EQ(typed_vals.size() / typed_keys.size(), chunk_size_);
    for (size_t index = 0; index < typed_keys.size(); index++) {
      CHECK_GE(typed_keys[index] * chunk_size_, range_.begin());
      CHECK_LT(typed_keys[index] * chunk_size_, range_.end());
      for (size_t chunk_index = 0; chunk_index < chunk_size_; chunk_index++)
        storage_[typed_keys[index] * chunk_size_ - range_.begin() + chunk_index] +=
            typed_vals[index * chunk_size_ + chunk_index];
    }
  }

  virtual third_party::SArray<char> SubGet(const third_party::SArray<Key>& typed_keys) override {
    third_party::SArray<Val> reply_vals(typed_keys.size());
    for (size_t i = 0; i < typed_keys.size(); i++) {
      CHECK_GE(typed_keys[i], range_.begin());
      CHECK_LT(typed_keys[i], range_.end());
      reply_vals[i] = storage_[typed_keys[i] - range_.begin()];
    }
    return third_party::SArray<char>(reply_vals);
  }

  virtual third_party::SArray<char> SubGetChunk(const third_party::SArray<Key>& typed_keys) override {
    third_party::SArray<Val> reply_vals(typed_keys.size() * chunk_size_);
    for (int i = 0; i < typed_keys.size(); ++i) {
      CHECK_GE(typed_keys[i] * chunk_size_, range_.begin());
      CHECK_LT(typed_keys[i] * chunk_size_, range_.end());
      for (int j = 0; j < chunk_size_; ++j)
        reply_vals[i * chunk_size_ + j] = storage_[typed_keys[i] * chunk_size_ - range_.begin() + j];
    }
    return third_party::SArray<char>(reply_vals);
  }

  virtual void FinishIter() override {}

  virtual void Clear() override {
    storage_.clear();
    storage_.resize(range_.size());
  }

  int GetBegin() { return range_.begin(); }

  int GetEnd() { return range_.end(); }

  size_t Size() const {
    CHECK_EQ(range_.size(), storage_.size());
    return storage_.size();
  }

  virtual void WriteTo(const std::string& file_path) override {
    std::ofstream file(file_path, std::ios::binary | std::ios::out);
    CHECK(file.is_open()) << "Cannot open file " << file_path;

    uint32_t range_begin = range_.begin(), range_end = range_.end();
    file.write((char*) &chunk_size_, sizeof(chunk_size_));
    file.write((char*) &range_begin, sizeof(uint32_t));
    file.write((char*) &range_end, sizeof(uint32_t));
    size_t storage_size = storage_.size();
    file.write((char*) &storage_size, sizeof(size_t));
    file.write((char*) storage_.data(), storage_size * sizeof(Val));
    file.flush();
    file.close();
  }

  virtual void LoadFrom(const std::string& file_path) override {
    std::ifstream infile(file_path, std::ios::binary | std::ios::in);
    CHECK(infile.is_open()) << "Cannot open file " << file_path;

    infile.read((char*) &chunk_size_, sizeof(chunk_size_));

    uint32_t range_begin = range_.begin(), range_end = range_.end();
    infile.read((char*) &range_begin, sizeof(uint32_t));
    infile.read((char*) &range_end, sizeof(uint32_t));
    range_ = third_party::Range(range_begin, range_end);

    size_t storage_size;
    infile.read((char*) &storage_size, sizeof(size_t));
    storage_.resize(storage_size);
    for (auto& val : storage_) {
      infile.read((char*) &val, sizeof(Val));
    }
    infile.close();
  }

 private:
  third_party::Range range_;
  std::vector<Val> storage_;
  uint32_t chunk_size_;
};

}  // namespace flexps
