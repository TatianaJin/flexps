#pragma once

#include "base/message.hpp"
#include "server/abstract_storage.hpp"

#include "glog/logging.h"

#include <fstream>
#include <map>

namespace flexps {

template <typename Val>
class MapStorage : public AbstractStorage {
 public:
  MapStorage(uint32_t chunk_size = 1) : chunk_size_(chunk_size) {}

  virtual void SubAdd(const third_party::SArray<Key>& typed_keys, const third_party::SArray<char>& vals) override {
    auto typed_vals = third_party::SArray<Val>(vals);
    for (size_t i = 0; i < typed_keys.size(); i++)
      storage_[typed_keys[i]] += typed_vals[i];
  }

  virtual void SubAddChunk(const third_party::SArray<Key>& typed_keys, const third_party::SArray<char>& vals) override {
    auto typed_vals = third_party::SArray<Val>(vals);
    CHECK_EQ(typed_vals.size() / typed_keys.size(), chunk_size_);
    for (size_t i = 0; i < typed_keys.size(); i++)
      for (size_t j = 0; j < chunk_size_; j++)
        storage_[typed_keys[i] * chunk_size_ + j] += typed_vals[i * chunk_size_ + j];
  }

  virtual third_party::SArray<char> SubGet(const third_party::SArray<Key>& typed_keys) override {
    third_party::SArray<Val> reply_vals(typed_keys.size());
    for (size_t i = 0; i < typed_keys.size(); i++)
      reply_vals[i] = storage_[typed_keys[i]];
    return third_party::SArray<char>(reply_vals);
  }

  virtual third_party::SArray<char> SubGetChunk(const third_party::SArray<Key>& typed_keys) override {
    third_party::SArray<Val> reply_vals(typed_keys.size() * chunk_size_);
    for (int i = 0; i < typed_keys.size(); i++)
      for (int j = 0; j < chunk_size_; j++)
        reply_vals[i * chunk_size_ + j] = storage_[typed_keys[i] * chunk_size_ + j];
    return third_party::SArray<char>(reply_vals);
  }

  virtual void FinishIter() override {}

  virtual void Clear() override { storage_.clear(); }

  virtual std::pair<third_party::SArray<Key>, third_party::SArray<Val>> GetKeysVals() {
    third_party::SArray<Key> keys;
    third_party::SArray<Val> vals;
    auto size = storage_.size();
    keys.reserve(size);
    vals.reserve(size);

    for (auto& pair : storage_) {
      keys.push_back(pair.first);
      vals.push_back(pair.second);
    }

    return std::make_pair(keys, vals);
  }

  virtual void WriteTo(const std::string& file_path) override {  // TODO(tatiana): unit test
    std::ofstream file(file_path, std::ios::binary | std::ios::out);
    CHECK(file.is_open()) << "Cannot open file " << file_path;

    file.write((char*) &chunk_size_, sizeof(chunk_size_));

    size_t storage_size = storage_.size();
    file.write((char*) &storage_size, sizeof(size_t));
    for (auto& key_val : storage_) {
      file.write((char*) &key_val.first, sizeof(Key));
      file.write((char*) &key_val.second, sizeof(Val));
    }

    file.flush();
    file.close();
  }

  virtual void LoadFrom(const std::string& file_path) override {
    std::ifstream file(file_path, std::ios::binary | std::ios::in);
    CHECK(file.is_open()) << "Cannot open file " << file_path;

    file.read((char*) &chunk_size_, sizeof(chunk_size_));

    size_t storage_size;
    file.read((char*) &storage_size, sizeof(size_t));
    storage_.clear();

    for (int i = 0; i < storage_size; ++i) {
      Key key;
      Val val;
      file.read((char*) &key, sizeof(Key));
      file.read((char*) &val, sizeof(Val));
      storage_[key] = val;
    }
    file.close();
  }

 protected:
  std::map<Key, Val> storage_;
  uint32_t chunk_size_;
};

}  // namespace flexps
