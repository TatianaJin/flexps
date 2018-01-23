#pragma once

#include <cmath>
#include <numeric>
#include <vector>

#include "glog/logging.h"

#include "base/magic.hpp"
#include "lib/data_loader/typed_labeled_sample.hpp"

namespace flexps {
namespace lib {

class Objective {
 public:
  using LabeledSample = TypedLabeledSample<ValT, ValT>;  // TODO(tatiana): use template?

  /**
   * Constructor
   *
   * @param num_params  The total number of parameters (including bias)
   */
  explicit Objective(int num_dims);

  inline int GetNumParams() const { return num_params_; }
  inline int GetNumFeatures() const { return num_dims_; }

  /**
   * Calculate averaged gradients using the batch of samples
   *
   * @param batch       a batch of data samples
   * @param params      the model parameters covering all non-zero features in the batch
   * @param keys        the keys of the corresponding model parameters
   * @param cardinality the denominator for calculating the average of gradients, equal to batch size if given 0
   *
   * @param delta       the returned gradients
   */
  virtual void GetGradient(const std::vector<LabeledSample*>& batch, const third_party::SArray<Key>& keys,
                           const third_party::SArray<ValT>& params, third_party::SArray<ValT>* delta,
                           int cardinality = 0) = 0;

  /**
   * Calculate averaged loss on the given samples
   *
   * @param samples     data samples
   * @param model       the model parameters covering all non-zero features in the given samples
   * @param keys        the keys of the corresponding model parameters. Not required when given the whole model
   * @param denominator the denominator for calculating the average of gradients, equal to batch size if given 0
   *
   * @returns the averaged loss
   */
  virtual ValT GetLoss(const std::vector<LabeledSample*>& samples, const third_party::SArray<ValT>& model,
                       ValT denominator = 0, const third_party::SArray<Key>* keys = nullptr) = 0;

  /**
   * Process keys for Get operation
   */
  virtual void ProcessKeys(third_party::SArray<Key>* keys);

  /**
   * Output all keys in the model
   */
  virtual void AllKeys(third_party::SArray<Key>* keys);

  /**
   * Predict the label of the given sample
   *
   * @param sample      A data sample
   * @param params      The model parameters covering all non-zero features in the given sample
   * @param keys        The keys of the corresponding model parameters. Not required when given the whole model
   *
   * @returns the predicted label
   */
  virtual ValT Predict(const LabeledSample& sample, const third_party::SArray<ValT>& params,
                       const third_party::SArray<Key>* keys = nullptr) = 0;

  /**
   * Calculate averaged accuracy on the given samples
   *
   * @param samples     Data samples
   * @param model       The model parameters covering all non-zero features in the given samples
   * @param keys        The keys of the corresponding model parameters. Not required when given the whole model
   * @param denominator The denominator for calculating the average of gradients, equal to batch size if given 0
   *
   * @returns the averaged accuracy
   */
  virtual ValT GetAccuracy(const std::vector<LabeledSample*>& samples, const third_party::SArray<ValT>& model,
                           const third_party::SArray<Key>* keys = nullptr, ValT denominator = 0) = 0;

  // std::vector API
  virtual void ProcessKeys(std::vector<Key>* keys);
  virtual void AllKeys(std::vector<Key>* keys);

 protected:
  int num_params_ = 0;
  int num_dims_ = 0;
};

}  // namespace lib
}  // namespace husky
