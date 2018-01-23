#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "base/node.hpp"
#include "base/node_util.hpp"
#include "comm/mailbox.hpp"
#include "driver/kv_engine.hpp"
#include "driver/simple_id_mapper.hpp"

namespace flexps {

class Engine {
 public:
  Engine(const Node& node, const std::vector<Node>& nodes) : node_(node), nodes_(nodes) {}

  void StartEverything(int num_server_threads_per_node = 1);

  void StopEverything();

  void Barrier();

  /*
   * TODO(tatiana) model_id management
   * Should be more flexible with customizable partitioning scheme.
   * Range partition can be used as default with simpler interface.
   * Flexibility: Use table info to create table, and consider string config/ or table info base
   */
  template <typename Val>
  void CreateTable(uint32_t table_id, const std::vector<third_party::Range>& ranges, ModelType model_type,
                   StorageType storage_type, int model_staleness = 0, uint32_t chunk_size = 1,
                   int dump_interval = 10000);

  /**
   * Create the partitions of a model on the local servers
   * 1. Assign a table id (incremental and consecutive)
   *
   * @param partition_manager   the model partition manager
   * @param model_type          the consistency of model - bsp, ssp, asp
   * @param storage_type        the storage type - map, vector...
   * @param model_staleness     the staleness for ssp model
   * @param chunk_size          the number of chunks of model parameters
   *
   * @return                    the created table(model) id
   * TODO(tatiana): unit test
   */
  template <typename Val>
  uint32_t CreateTable(std::unique_ptr<AbstractPartitionManager> partition_manager, ModelType model_type,
                       StorageType storage_type, int model_staleness = 0, uint32_t chunk_size = 1,
                       int dump_interval = 10000);

  /**
   * Create the partitions of a model on the local servers using a default range partitioning scheme
   * 1. Create a range partition manager
   * 2. Create a table with the partition manager
   *
   * @param model_type          the consistency of model - bsp, ssp, asp
   * @param storage_type        the storage type - map, vector...
   * @param n_keys              the number of parameters in the table
   * @param model_staleness     the staleness for ssp model
   *
   * @return                    the created table(model) id
   * TODO(tatiana): unit test
   */
  template <typename Val>
  uint32_t CreateTable(ModelType model_type, StorageType storage_type, int n_keys = 10, int model_staleness = 0,
                       uint32_t chunk_size = 1, int dump_interval = 10000);

  void Run(const MLTask& task);

  SimpleIdMapper* GetIdMapper() {
    CHECK(id_mapper_);
    return id_mapper_.get();
  }

  Mailbox* GetMailbox() {
    CHECK(mailbox_);
    return mailbox_.get();
  }

  // For dev use only
  // Create SparseSSP Table, for testing sparsessp use only.
  // Make sure you know how to use sparsessp before use this.
  template <typename Val>
  void CreateSparseSSPTable(uint32_t table_id, const std::vector<third_party::Range>& ranges, ModelType model_type,
                            StorageType storage_type, int model_staleness = 0, int speculation = 0,
                            SparseSSPRecorderType sparse_ssp_recorder_type = SparseSSPRecorderType::None);

 private:
  // nodes
  Node node_;
  std::vector<Node> nodes_;

  std::unique_ptr<SimpleIdMapper> id_mapper_;
  std::unique_ptr<Mailbox> mailbox_;
  std::unique_ptr<KVEngine> kv_engine_;
  size_t model_count_ = 0;
};

template <typename Val>
void Engine::CreateTable(uint32_t table_id, const std::vector<third_party::Range>& ranges, ModelType model_type,
                         StorageType storage_type, int model_staleness, uint32_t chunk_size, int dump_interval) {
  CHECK(kv_engine_);
  model_count_ = table_id + 1;
  kv_engine_->CreateTable<Val>(table_id, ranges, model_type, storage_type, model_staleness, chunk_size, dump_interval);
}

template <typename Val>
uint32_t Engine::CreateTable(std::unique_ptr<AbstractPartitionManager> partition_manager, ModelType model_type,
                             StorageType storage_type, int model_staleness, uint32_t chunk_size, int dump_interval) {
  uint32_t model_id = model_count_++;
  kv_engine_->CreateTable<Val>(model_id, std::move(partition_manager), model_type, storage_type, model_staleness,
                               chunk_size, dump_interval);
  return model_id;
}

template <typename Val>
uint32_t Engine::CreateTable(ModelType model_type, StorageType storage_type, int n_keys, int model_staleness,
                             uint32_t chunk_size, int dump_interval) {
  CHECK(id_mapper_);
  CHECK(kv_engine_);

  auto server_ids = id_mapper_->GetAllServerThreads();
  int num_server_threads = server_ids.size();

  // Divide n_keys into ranges
  std::vector<third_party::Range> ranges;
  ranges.reserve(num_server_threads);
  auto remainder = n_keys % num_server_threads;
  auto division = n_keys / num_server_threads;
  for (Key i = 0; i < remainder; ++i) {
    int size = division + 1;
    ranges.push_back({i * size, (i + 1) * size});
  }

  for (Key i = remainder; i < num_server_threads; ++i) {
    ranges.push_back({remainder + i * division, remainder + (i + 1) * division});
  }

  kv_engine_->CreateTable<Val>(model_count_++, ranges, model_type, storage_type, model_staleness, chunk_size,
                               dump_interval);
  return model_count_ - 1;
}

template <typename Val>
void Engine::CreateSparseSSPTable(uint32_t table_id, const std::vector<third_party::Range>& ranges,
                                  ModelType model_type, StorageType storage_type, int model_staleness, int speculation,
                                  SparseSSPRecorderType sparse_ssp_recorder_type) {
  CHECK(kv_engine_);
  kv_engine_->CreateSparseSSPTable<Val>(table_id, ranges, model_type, storage_type, model_staleness, speculation,
                                        sparse_ssp_recorder_type);
}

}  // namespace flexps
