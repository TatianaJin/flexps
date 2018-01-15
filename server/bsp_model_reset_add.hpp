#pragma once

#include "server/abstract_model.hpp"

#include "base/message.hpp"
#include "base/threadsafe_queue.hpp"
#include "server/abstract_storage.hpp"
#include "server/bsp_model.hpp"
#include "server/pending_buffer.hpp"
#include "server/progress_tracker.hpp"

#include <map>
#include <vector>

namespace flexps {

class BSPModelResetAdd : public BSPModel {
 public:
  BSPModelResetAdd(uint32_t model_id, std::unique_ptr<AbstractStorage>&& storage_ptr,
                   ThreadsafeQueue<Message>* reply_queue);

  virtual void Clock(Message& msg) override;
};

}  // namespace flexps
