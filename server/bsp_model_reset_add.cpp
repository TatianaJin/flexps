#include "server/bsp_model_reset_add.hpp"

#include "glog/logging.h"
namespace flexps {

BSPModelResetAdd::BSPModelResetAdd(uint32_t model_id, std::unique_ptr<AbstractStorage>&& storage_ptr,
                                   ThreadsafeQueue<Message>* reply_queue, int dump_interval)
    : BSPModel(model_id, std::move(storage_ptr), reply_queue, dump_interval) {}

void BSPModelResetAdd::Clock(Message& msg) {
  int updated_min_clock = progress_tracker_.AdvanceAndGetChangedMinClock(msg.meta.sender);
  int progress = progress_tracker_.GetProgress(msg.meta.sender);
  CHECK_LE(progress, progress_tracker_.GetMinClock() + 1);
  if (updated_min_clock != -1) {  // min clock updated
    storage_->Clear();            // reset in each iteration before add

    for (auto add_req : add_buffer_) {
      storage_->Add(add_req);
    }
    add_buffer_.clear();

    storage_->FinishIter();
    if (dump_interval_ > 0 && updated_min_clock % dump_interval_ == 0) {
      LOG(INFO) << "[BSPModelResetAdd] Version, Timestamp: " << updated_min_clock << ","
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                       .count();
      Dump(server_id_);
    }

    for (auto get_req : get_buffer_) {
      reply_queue_->Push(storage_->Get(get_req));
    }
    get_buffer_.clear();
  }
}

}  // namespace flexps
