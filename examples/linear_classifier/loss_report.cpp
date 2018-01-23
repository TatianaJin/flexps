#include "gflags/gflags.h"
#include "glog/logging.h"

#include <iomanip>

#include "driver/engine.hpp"
#include "lib/data_loader/data_store.hpp"
#include "lib/data_loader/hdfs_data_loader.hpp"
#include "lib/data_loader/parser.hpp"
#include "lib/objectives/linear_regression_objective.hpp"
#include "lib/objectives/objective.hpp"
#include "lib/objectives/sigmoid_objective.hpp"
#include "lib/regularizer.hpp"
#include "worker/kv_client_table.hpp"

DEFINE_int32(my_id, -1, "The process id of this program");
DEFINE_string(config_file, "", "The config file path");
DEFINE_string(hdfs_namenode, "", "The hdfs namenode hostname");
DEFINE_int32(hdfs_namenode_port, -1, "The hdfs namenode port");
DEFINE_int32(num_workers_per_node, 1, "num_workers_per_node");
DEFINE_int32(num_servers_per_node, 1, "num_servers_per_node");

DEFINE_string(input, "", "The hdfs input url");
DEFINE_string(model_input, "", "The model input path, each node load one partition for each version");
DEFINE_int32(num_dims, 0, "number of dimensions");
DEFINE_int32(cardinality, 0, "The number of samples");
DEFINE_int32(report_interval, 0, "model dump interval");
DEFINE_int32(max_version, 0, "inclusive max model version");

DEFINE_string(kStorageType, "", "Map/Vector");
DEFINE_string(trainer, "logistic", "logistic|linear");

DEFINE_string(regularizer, "none", "none|l1|l2|elastic_net");
DEFINE_double(eta1, 0.001, "l1 regularization factor");
DEFINE_double(eta2, 0.001, "l2 regularization factor");

using namespace flexps;
using lib::HDFSDataLoader;
using lib::Objective;
using lib::SigmoidObjective;
using lib::LinearRegressionObjective;

// DataObj = <feature<key, val>, label>
using DataObj = lib::Objective::LabeledSample;
using DataStore = lib::DataStore<DataObj>;
using BatchIterator = lib::BatchIterator<DataObj>;
using Parser = lib::Parser<DataObj>;

third_party::SArray<Key> GetKeys(std::vector<DataObj*> samples) {
  std::set<Key> key_set;
  for (auto* sample : samples) {
    auto& x = sample->x_;
    for (auto& key_val : x) {
      key_set.insert(key_val.first);
    }
  }
  third_party::SArray<Key> keys;
  keys.reserve(key_set.size());
  for (auto key : key_set) {
    keys.push_back(key);
  }
  return keys;
}

void Run() {
  CHECK_NE(FLAGS_my_id, -1) << "[LRLossReport] my_id not specified";
  CHECK(!FLAGS_config_file.empty()) << "[LRLossReport] node config file not specified";
  VLOG(1) << FLAGS_my_id << " " << FLAGS_config_file;

  // 0. Parse config_file
  std::vector<Node> nodes = ParseFile(FLAGS_config_file);
  CHECK(CheckValidNodeIds(nodes));
  CHECK(CheckUniquePort(nodes));
  CHECK(CheckConsecutiveIds(nodes));
  Node my_node = GetNodeById(nodes, FLAGS_my_id);
  LOG(INFO) << my_node.DebugString();

  // 1. Load data
  DataStore data_store(FLAGS_num_workers_per_node);
  auto loader = HDFSDataLoader::Get(my_node, FLAGS_hdfs_namenode, FLAGS_hdfs_namenode_port, nodes[0].hostname, 20954,
                                    nodes.size());
  loader->Load(FLAGS_input, FLAGS_num_dims, Parser::parse_libsvm, &data_store, FLAGS_num_workers_per_node);

  // 2. Start engine
  Engine engine(my_node, nodes);
  engine.StartEverything();

  // 3. Create tables
  ModelType model_type = ModelType::BSPResetAdd;
  StorageType storage_type;
  if (FLAGS_kStorageType == "Map") {
    storage_type = StorageType::Map;
  } else if (FLAGS_kStorageType == "Vector") {
    storage_type = StorageType::Vector;
  } else {
    CHECK(false) << "storage type error: " << FLAGS_kStorageType;
  }

  int num_params = FLAGS_num_dims + 1;
  auto table_id = engine.CreateTable<ValT>(model_type, storage_type, num_params, 1, 0);
  auto loss_table_id = engine.CreateTable<ValT>(model_type, storage_type, 2, 1, 0);
  engine.Barrier();

  // 3. Specify training algoritm
  // a. Set objective
  std::shared_ptr<Objective> objective_ptr;
  if (FLAGS_trainer == "logistic") {
    objective_ptr = std::make_shared<SigmoidObjective>(FLAGS_num_dims);
  } else if (FLAGS_trainer == "linear") {
    objective_ptr = std::make_shared<LinearRegressionObjective>(FLAGS_num_dims);
  } else {
    LOG(ERROR) << "Trainer type not supported";
  }
  // b. Set regularization
  int regularizer = 0;  // default 0
  if (FLAGS_regularizer == "l1") {
    regularizer = 1;
  } else if (FLAGS_regularizer == "l2") {
    regularizer = 2;
  } else if (FLAGS_regularizer == "elastic_net") {
    regularizer = 3;
  }

  // 3. Construct tasks
  MLTask task;
  std::vector<WorkerAlloc> worker_alloc;
  for (auto& node : nodes) {
    worker_alloc.push_back(
        {node.id, static_cast<uint32_t>(FLAGS_num_workers_per_node)});  // each node has num_workers_per_node workers
  }
  task.SetWorkerAlloc(worker_alloc);
  task.SetTables({table_id, loss_table_id});  // Use table 0

  task.SetLambda([storage_type, table_id, loss_table_id, regularizer, objective_ptr, &data_store](const Info& info) {
    VLOG(1) << info.DebugString();

    auto table = info.CreateKVClientTable<ValT>(table_id);
    auto loss_table = info.CreateKVClientTable<ValT>(loss_table_id);

    auto samples = data_store.GetPtrs(info.local_id);
    third_party::SArray<Key> keys = GetKeys(samples);
    objective_ptr->ProcessKeys(&keys);
    third_party::SArray<ValT> model(keys.size(), 0.0f);  // parameters for local samples

    third_party::SArray<Key> key_part;
    third_party::SArray<ValT> model_part;
    // Compute loss for each model version
    for (int version = 0; version <= FLAGS_max_version; version += FLAGS_report_interval) {
      if (version > 0) {
        // Get model parameters
        if (info.local_id == 0) {
          // a. Load model partition
          if (storage_type == StorageType::Vector) {
            VectorStorage<ValT> storage({0, 1});
            storage.LoadFrom(FLAGS_model_input + "MODEL_v" + std::to_string(version) + "_part0");
            auto first_key = storage.GetBegin();
            auto last_key = storage.GetEnd();
            key_part.clear();
            key_part.reserve(storage.Size());
            for (auto key = first_key; key < last_key; ++key) {
              key_part.push_back(key);
            }
            model_part = third_party::SArray<ValT>(storage.SubGet(key_part));
          } else if (storage_type == StorageType::Map) {
            MapStorage<ValT> storage;
            storage.LoadFrom(FLAGS_model_input + "MODEL_v" + std::to_string(version) + "_part" +
                             std::to_string(FLAGS_my_id));
            auto keys_vals = storage.GetKeysVals();
            key_part = std::move(keys_vals.first);
            model_part = std::move(keys_vals.second);
          }

          // b. Update to parameter servers
          table.Add(key_part, model_part);
        }
        table.Clock();
        if (keys.empty()) {  // To avoid problem with too advanced clock in BSP
          keys.push_back(0);
        }
        table.Get(keys, &model);
      }

      // 2. Calculate loss and add regularization penalty distributedly by local leaders
      ValT loss = objective_ptr->GetLoss(samples, model, FLAGS_cardinality, &keys);
      ValT accuracy = objective_ptr->GetAccuracy(samples, model, &keys, FLAGS_cardinality);

      if (info.local_id == 0) {
        for (auto param : model_part) {
          loss += lib::regularization_penalty(regularizer, param, FLAGS_eta1, FLAGS_eta2);
        }
      }

      // 3. Update loss and output
      third_party::SArray<ValT> loss_arr({loss, accuracy});
      loss_table.Add({0, 1}, loss_arr);
      loss_table.Clock();
      loss_table.Get({0, 1}, &loss_arr);
      if (info.worker_id == 0) {
        CHECK_EQ(loss_arr.size(), 2);
        LOG(INFO) << "Iteration, loss, accuracy: " << version << std::setprecision(15) << "," << loss_arr[0] << ","
                  << loss_arr[1];
      }
    }
  });

  // 4. Run tasks
  engine.Run(task);
  // 5. Stop engine
  engine.StopEverything();
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  Run();
}
