#include "server/abstract_model.hpp"

#include "base/message.hpp"
#include "base/threadsafe_queue.hpp"
#include "server/abstract_storage.hpp"
#include "server/pending_buffer.hpp"
#include "server/progress_tracker.hpp"

#include <map>
#include <vector>

namespace flexps {

class SSPModel : public AbstractModel {
 public:
  SSPModel(uint32_t model_id, std::unique_ptr<AbstractStorage>&& storage_ptr, int staleness,
           ThreadsafeQueue<Message>* reply_queue, int dump_interval = 10000);

  virtual void Clock(Message& msg) override;
  virtual void Add(Message& msg) override;
  virtual void Get(Message& msg) override;
  virtual int GetProgress(int tid) override;
  virtual void ResetWorker(Message& msg) override;

  int GetPendingSize(int progress);

  virtual void Dump(int server_id, const std::string& path = "") override;
  virtual void Load(const std::string& file_name) override;

 private:
  uint32_t model_id_;
  uint32_t staleness_;

  ThreadsafeQueue<Message>* reply_queue_;
  std::unique_ptr<AbstractStorage> storage_;
  ProgressTracker progress_tracker_;
  PendingBuffer buffer_;
};

}  // namespace flexps
