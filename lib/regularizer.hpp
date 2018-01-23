#pragma once

#include <cmath>

#include "base/magic.hpp"

namespace flexps {
namespace lib {

/**
 * Proximal regularizer
 *
 * Supports L1, L2 and Elastic net regularization
 *
 * @param regular   The type of regularization, 1 for l1, 2 for l2, 3 for elastic net
 * @param prox      The parameter to regularize
 * @param step_size Learning rate
 * @param lambda1   The l1 regularization factor
 * @param lambda2   The l2 regularization factor
 *
 * @returns         The regularized parameter
 */
using PrecisionT = ValT;

PrecisionT regularization_penalty(int regular, PrecisionT param, PrecisionT lambda1, PrecisionT lambda2) {
  PrecisionT penalty = 0;
  if (regular == 1 || regular == 3) {
    penalty += lambda1 * abs(param);
  }
  if (regular == 2 || regular == 3) {
    penalty += 0.5 * lambda2 * param * param;
  }
  return penalty;
}

PrecisionT proximal_operator(int regular, PrecisionT& prox, PrecisionT step_size, PrecisionT lambda1,
                             PrecisionT lambda2) {
  switch (regular) {
  case 1: {
    PrecisionT param = step_size * lambda1;
    if (prox > param)
      prox -= param;
    else if (prox < -param)
      prox += param;
    else
      prox = 0;
    return prox;
  }
  case 2: {
    prox /= (1 + step_size * lambda2);
    return prox;
  }
  case 3: {
    PrecisionT param_1 = step_size * lambda1;
    PrecisionT param_2 = 1.0 / (1.0 + step_size * lambda2);
    if (prox > param_1)
      prox = param_2 * (prox - param_1);
    else if (prox < -param_1)
      prox = param_2 * (prox + param_1);
    else
      prox = 0;
    return prox;
  }
  default:  // no regularization
    return prox;
  }
}

}  // namespace lib
}  // namespace flexps
