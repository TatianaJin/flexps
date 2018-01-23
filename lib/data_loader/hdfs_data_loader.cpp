#include "lib/data_loader/hdfs_data_loader.hpp"

#include <functional>
#include <string>
#include <thread>

#include "boost/utility/string_ref.hpp"
#include "glog/logging.h"

#include "base/node.hpp"
#include "driver/info.hpp"
#include "io/hdfs_assigner.hpp"
#include "io/lineinput.hpp"
#include "lib/data_loader/parser.hpp"

namespace flexps {
namespace lib {

HDFSDataLoader::Config::Config(const Node& _node, const std::string& _hdfs_namenode, int _hdfs_namenode_port,
                               const std::string& _master_host, int _master_port, int _n_nodes)
    : node(_node),
      hdfs_namenode(_hdfs_namenode),
      hdfs_namenode_port(_hdfs_namenode_port),
      master_host(_master_host),
      master_port(_master_port),
      n_nodes(_n_nodes) {}

HDFSDataLoader::~HDFSDataLoader() { StopMaster(); }

void HDFSDataLoader::StartMaster() {
  if (config_.node.hostname != config_.master_host || master_start_)
    return;
  hdfs_main_thread_ = std::thread([this] {
    HDFSBlockAssigner hdfs_block_assigner(config_.hdfs_namenode, config_.hdfs_namenode_port, &context_,
                                          config_.master_port);
    hdfs_block_assigner.Serve();
  });
  master_start_ = true;
}

void HDFSDataLoader::StopMaster() {
  if (config_.node.hostname == config_.master_host && master_start_) {
    master_start_ = false;
    hdfs_main_thread_.join();
  }
}

std::unique_ptr<HDFSDataLoader> HDFSDataLoader::Get(const Node& node, const std::string& hdfs_namenode,
                                                    int hdfs_namenode_port, const std::string& master_host,
                                                    int master_port, int n_nodes, bool start_master) {
  static int loader_count_ = 0;
  Config config(node, hdfs_namenode, hdfs_namenode_port, master_host, master_port, n_nodes);
  return std::unique_ptr<HDFSDataLoader>(new HDFSDataLoader(config, loader_count_++, start_master));
}

HDFSDataLoader::HDFSDataLoader(const Config& config, int task_idx, bool start_master)
    : config_(config), task_idx_(task_idx) {
  context_ = zmq::context_t(1);
  if (start_master) {
    StartMaster();
  }
  coordinator_.reset(
      new Coordinator(config.node.id, config.node.hostname, &context_, config.master_host, config.master_port));
  coordinator_->serve();
}

LineInputFormat HDFSDataLoader::CreateLineInputFormat(const std::string& url, int tid, int n_loading_threads) const {
  CHECK(coordinator_);
  return LineInputFormat(url, n_loading_threads, tid, coordinator_.get(), config_.node.hostname, config_.hdfs_namenode,
                         config_.hdfs_namenode_port);
}

}  // namespace lib
}  // namespace flexps
