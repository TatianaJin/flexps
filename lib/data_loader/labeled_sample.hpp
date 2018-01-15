#pragma once

#include "base/magic.hpp"
#include "base/third_party/sarray.h"

namespace flexps {
namespace lib {

template <typename Feature, typename Label>
class TypedLabeledSample {
 public:
  TypedLabeledSample(int n_features) {}

  std::vector<std::pair<Key, Feature>> x_;
  Label y_;
};  // class TypedLabeledSample

}  // namespace lib
}  // namespace flexps
