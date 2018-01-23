#pragma once

#include <cinttypes>
#include "base/message.hpp"
#include "base/threadsafe_queue.hpp"

namespace flexps {

class AbstractModel {
 public:
  virtual void Clock(Message& msg) = 0;
  virtual void Add(Message& msg) = 0;
  virtual void Get(Message& msg) = 0;
  virtual int GetProgress(int tid) = 0;
  virtual void ResetWorker(Message& msg) = 0;
  virtual void Dump(int server_id, const std::string& path = ""){};
  virtual void Load(const std::string& file_name){};
  virtual ~AbstractModel() {}
  void SetServerId(int server_id) { server_id_ = server_id; }
  void SetDumpInterval(int dump_interval) { dump_interval_ = dump_interval; }

 protected:
  int server_id_ = 0;
  int dump_interval_ = 0;
};

}  // namespace flexps
