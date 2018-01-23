#pragma once

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

class HDFSDataLoader {
 public:
  struct Config {
    Config(const Node& _node, const std::string& _hdfs_namenode, int _hdfs_namenode_port,
           const std::string& _master_host, int _master_port, int _n_nodes);

    Node node;
    std::string hdfs_namenode;
    int hdfs_namenode_port;
    std::string master_host;
    int master_port;
    int n_nodes;
  };

  ~HDFSDataLoader();

  void StartMaster();
  void StopMaster();

  /**
   * Load samples from the url into datastore
   *
   * @param url          input file/directory
   * @param n_features   the number of features in the dataset
   * @param parse        a parsing function
   * @param datastore    a container for the samples / external in-memory storage abstraction
   */
  template <typename Parse, typename DataStore>  // e.g. std::function<Sample(boost::string_ref, int)>
  void Load(std::string url, int n_features, Parse parse, DataStore* datastore, int n_threads_per_node) {
    // 1. Connect to the data source, e.g. HDFS, via the modules in io
    std::vector<std::thread> threads;
    threads.reserve(n_threads_per_node);
    for (int tid = 0; tid < n_threads_per_node; ++tid) {
      threads.push_back(std::thread([n_features, tid, url, n_threads_per_node, &datastore, &parse, this] {
        int n_loading_threads = n_threads_per_node * config_.n_nodes;
        LineInputFormat infmt = CreateLineInputFormat(url, task_idx_, n_loading_threads);

        boost::string_ref record;
        bool success = false;
        // 2. Extract lines
        while (true) {
          success = infmt.next(record);
          if (success == false)
            break;
          // 3. Parse line and put samples into datastore
          datastore->Push(tid, parse(record, n_features));
        }

        BinStream finish_signal;
        finish_signal << config_.node.hostname << config_.node.id * n_threads_per_node + tid;
        coordinator_->notify_master(finish_signal, HDFSBlockAssigner::kExit);
      }));
    }

    for (auto& t : threads)
      t.join();
  }

  /**
   * Create an HDFSDataLoader
   *
   * @param node                the node info of the current process
   * @param hdfs_namenode       the hostname of the namenode of HDFS
   * @param hdfs_namenode_port  the port to connect to on the namenode of HDFS
   * @param master_host         the hostname of HDFS assigner
   * @param master_port         the port to connect to on the host of HDFS assigner
   * @param n_nodes             the number of nodes loading data
   * @return HDFSDataLoader
   */
  static std::unique_ptr<HDFSDataLoader> Get(const Node& node, const std::string& hdfs_namenode, int hdfs_namenode_port,
                                             const std::string& master_host, int master_port, int n_nodes,
                                             bool start_master = true);

 private:
  HDFSDataLoader(const Config& config, int task_idx, bool start_master);

  /**
   * @param n_loading_threads   the number of global threads loading data concurrently
   */
  LineInputFormat CreateLineInputFormat(const std::string& url, int tid, int n_loading_threads) const;

  Config config_;
  zmq::context_t context_;
  std::unique_ptr<Coordinator> coordinator_;
  std::thread hdfs_main_thread_;
  bool master_start_ = false;
  int task_idx_;

};  // class HDFSDataLoader

}  // namespace lib
}  // namespace flexps
