#pragma once

#include "boost/utility/string_ref.hpp"
#include "glog/logging.h"

#include "lib/data_loader/typed_labeled_sample.hpp"

namespace flexps {
namespace lib {

template <typename Sample>
class Parser {
 public:
  /**
   * Parsing logic for one line in file
   *
   * @param line    a line read from the input file
   */
  static Sample parse_libsvm(boost::string_ref line, int n_features) {
    Sample sample(n_features);
    char* pos;  // split position
    auto n_chars = line.size();
    std::unique_ptr<char> line_ptr(new char[n_chars + 1]);
    strncpy(line_ptr.get(), line.data(), n_chars);
    line_ptr.get()[n_chars] = '\0';

    // Parse label
    char* tok = strtok_r(line_ptr.get(), " \t:", &pos);
    CHECK(tok != NULL) << "[Parser::parse_libsvm] Invalid line format";
    sample.y_ = std::atof(tok);

    // Parse features
    tok = strtok_r(NULL, " \t:", &pos);
    while (tok != NULL) {
      Key idx = std::atoi(tok) - 1;
      tok = strtok_r(NULL, " \t:", &pos);
      CHECK(tok != NULL) << "[Parser::parse_libsvm] Invalid line format";
      sample.x_.push_back(std::make_pair(idx, std::atof(tok)));
      tok = strtok_r(NULL, " \t:", &pos);
    }

    return sample;
  }

};  // class Parser

}  // namespace lib
}  // namespace flexps
