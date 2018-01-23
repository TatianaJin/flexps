#include "lib/objectives/objective.hpp"

#include "base/third_party/sarray.h"

namespace flexps {
namespace lib {

Objective::Objective(int num_dims) : num_params_(num_dims + 1), num_dims_(num_dims) {}

void Objective::ProcessKeys(third_party::SArray<Key>* keys) {
  if (keys->empty()) {
    return;
  }
  if (keys->back() != num_params_ - 1)
    keys->push_back(num_params_ - 1);  // add key for bias
}

void Objective::AllKeys(third_party::SArray<Key>* keys) {
  keys->resize(num_params_);
  std::iota(keys->begin(), keys->end(), 0);
}

void Objective::ProcessKeys(std::vector<Key>* keys) {
  if (keys->empty()) {
    return;
  }
  if (keys->back() != num_params_ - 1)
    keys->push_back(num_params_ - 1);  // add key for bias
}

void Objective::AllKeys(std::vector<Key>* keys) {
  keys->resize(num_params_);
  std::iota(keys->begin(), keys->end(), 0);
}

}  // namespace lib
}  // namespace husky
