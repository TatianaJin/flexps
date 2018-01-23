#include "gflags/gflags.h"
#include "glog/logging.h"

// #include <gperftools/profiler.h>

#include "driver/engine.hpp"
#include "lib/data_loader/data_store.hpp"
#include "lib/data_loader/hdfs_data_loader.hpp"
#include "lib/data_loader/parser.hpp"
#include "lib/objectives/linear_regression_objective.hpp"
#include "lib/objectives/objective.hpp"
#include "lib/objectives/sigmoid_objective.hpp"
#include "lib/optimizers/optimizer.hpp"
#include "lib/optimizers/sgd_optimizer.hpp"
#include "lib/optimizers/svrg_optimizer.hpp"
#include "worker/kv_client_table.hpp"

DEFINE_int32(my_id, -1, "The process id of this program");
DEFINE_string(config_file, "", "The config file path");
DEFINE_string(hdfs_namenode, "", "The hdfs namenode hostname");
DEFINE_int32(hdfs_namenode_port, -1, "The hdfs namenode port");
DEFINE_int32(num_workers_per_node, 1, "num_workers_per_node");
DEFINE_int32(num_servers_per_node, 1, "num_servers_per_node");

DEFINE_string(input, "", "The hdfs input url");
DEFINE_int32(num_dims, 0, "number of dimensions");
DEFINE_int32(cardinality, 0, "The number of samples");
DEFINE_int32(report_interval, 0, "model dump interval");

DEFINE_string(kStorageType, "", "Map/Vector");
DEFINE_string(kModelType, "ASP", "ASP/SSP/BSP");
DEFINE_int32(kStaleness, 0, "staleness");
DEFINE_string(trainer, "logistic", "logistic|linear");
DEFINE_string(optimizer, "sgd", "sgd|svrg");
DEFINE_bool(async, true, "true|false");
DEFINE_int32(batch_size, 100, "batch size of each stochastic iteration");
DEFINE_int32(num_iters, 10, "number of iters");
DEFINE_int32(num_epoches, 5, "number of epoches for svrg");
DEFINE_double(alpha, 0.1, "learning rate");
DEFINE_string(regularizer, "none", "none|l1|l2|elastic_net");
DEFINE_double(eta1, 0.001, "l1 regularization factor");
DEFINE_double(eta2, 0.001, "l2 regularization factor");

using namespace flexps;
using lib::HDFSDataLoader;
using lib::Objective;
using lib::Optimizer;
using lib::OptimizerConfig;
using lib::SGDOptimizer;
using lib::SVRGOptimizer;
using lib::SigmoidObjective;
using lib::LinearRegressionObjective;

// DataObj = <feature<key, val>, label>
using DataObj = lib::Optimizer::LabeledSample;
using DataStore = lib::DataStore<DataObj>;
using BatchIterator = lib::BatchIterator<DataObj>;
using Parser = lib::Parser<DataObj>;

void Run() {
  CHECK_NE(FLAGS_my_id, -1);
  CHECK(!FLAGS_config_file.empty());
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

  auto samples = data_store.Get();
  LOG(INFO) << "Finished loading " << samples.size() << " records";
  int count = 0;
  for (int i = 0; i < 100; i++) {
    count += samples[i]->x_.size();
  }
  LOG(INFO) << "Estimated number of non-zero: " << count / 100;

  // 2. Start engine
  Engine engine(my_node, nodes);
  engine.StartEverything();

  // 3. Create tables
  ModelType model_type;
  if (FLAGS_kModelType == "ASP") {
    model_type = ModelType::ASP;
  } else if (FLAGS_kModelType == "SSP") {
    model_type = ModelType::SSP;
  } else if (FLAGS_kModelType == "BSP") {
    model_type = ModelType::BSP;
  } else {
    CHECK(false) << "model type error: " << FLAGS_kModelType;
  }
  StorageType storage_type;
  if (FLAGS_kStorageType == "Map") {
    storage_type = StorageType::Map;
  } else if (FLAGS_kStorageType == "Vector") {
    storage_type = StorageType::Vector;
  } else {
    CHECK(false) << "storage type error: " << FLAGS_kStorageType;
  }

  int num_params = FLAGS_num_dims + 1;
  auto table_id =
      engine.CreateTable<ValT>(model_type, storage_type, num_params, FLAGS_kStaleness, 1, FLAGS_report_interval);
  uint32_t g_table_id = -1;  // full gradient table
  if (FLAGS_optimizer == "svrg") {
    // BSP model, reset every iter
    g_table_id = engine.CreateTable<ValT>(ModelType::BSPResetAdd, storage_type, num_params, 0, 1, 0);
  }
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
  // b. Set optimizer
  lib::OptimizerConfig conf;
  std::shared_ptr<Optimizer> optimizer_ptr;
  if (FLAGS_optimizer == "sgd") {
    optimizer_ptr = std::make_shared<SGDOptimizer>(objective_ptr, 0);
  } else if (FLAGS_optimizer == "svrg") {
    optimizer_ptr = std::make_shared<SVRGOptimizer>(objective_ptr, 0);
    conf.num_epoches = FLAGS_num_epoches;
    conf.cardinality = FLAGS_cardinality;
    CHECK(conf.cardinality != 0) << "Cardinality must be given";
    conf.async = FLAGS_async;
    LOG(INFO) << "async: " << conf.async;
    conf.gradient_table_id = g_table_id;
  }
  // c. Set hyperparameters
  conf.batch_size = FLAGS_batch_size;
  conf.num_iters = FLAGS_num_iters;
  conf.alpha = FLAGS_alpha;
  conf.learning_rate_decay = conf.num_iters;  // no decay
  conf.eta1 = FLAGS_eta1;
  conf.eta2 = FLAGS_eta2;
  if (FLAGS_regularizer == "l1") {
    conf.regularizer = 1;
  } else if (FLAGS_regularizer == "l2") {
    conf.regularizer = 2;
  } else if (FLAGS_regularizer == "elastic_net") {
    conf.regularizer = 3;
  }  // default 0

  // 3. Construct tasks
  MLTask task;
  std::vector<WorkerAlloc> worker_alloc;
  for (auto& node : nodes) {
    worker_alloc.push_back(
        {node.id, static_cast<uint32_t>(FLAGS_num_workers_per_node)});  // each node has num_workers_per_node workers
  }
  task.SetWorkerAlloc(worker_alloc);
  if (FLAGS_optimizer == "svrg") {
    task.SetTables({table_id, g_table_id});  // Use table 0
  } else
    task.SetTables({table_id});  // Use table 0

  task.SetLambda([table_id, optimizer_ptr, &conf, &data_store](const Info& info) {
    LOG(INFO) << info.DebugString();

    auto start_time = std::chrono::steady_clock::now();
    optimizer_ptr->Train(info, table_id, data_store, conf);
    auto end_time = std::chrono::steady_clock::now();

    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
    LOG(INFO) << "total time: " << total_time << " ms on worker: " << info.worker_id;
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
