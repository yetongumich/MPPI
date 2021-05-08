/* ----------------------------------------------------------------------------
 * GTDynamics Copyright 2020, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 *  @file testCartpoleFactors.cpp
 *  @brief Tests for cartpole factors.
 *  @author Yetong Zhang
 **/

#include <CppUnitLite/TestHarness.h>
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Testable.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/factorTesting.h>
#include <gtsam/slam/PriorFactor.h>

#include <iostream>

#include "gtdynamics/jumpingrobot/factors/CartpoleFactors.h"

using gtdynamics::CPDynamicsFactor, gtdynamics::CPStateCostFactor;
using gtsam::Symbol, gtsam::Vector5, gtsam::Vector4, gtsam::Values, gtsam::Key,
    gtsam::assert_equal, gtsam::noiseModel::Isotropic;

namespace example {
auto cost_model = Isotropic::Sigma(5, 1);
auto cost_model1 = Isotropic::Sigma(4, 0.001);
gtsam::Symbol x1_key('x', 0), x2_key('x', 1);
}  // namespace example

TEST(CPDynamicsFactor, Factor) {
  double dt = 0.02;
  Vector5 x1, x2;
  x1 << 1, 1, M_PI/6, 1, 1;
  x2 << 1.02, 1.02082251, M_PI/6+0.02, 0.53546869, 0.6;

  CPDynamicsFactor factor(example::x1_key, example::x2_key, example::cost_model, dt);

  Vector5 actual_errors, expected_errors;
  actual_errors = factor.evaluateError(x1, x2);
  expected_errors << 0, 0, 0, 0, 0;

  EXPECT(assert_equal(expected_errors, actual_errors, 1e-5));
  // Make sure linearization is correct
  Values values;
  values.insert(example::x1_key, x1);
  values.insert(example::x2_key, x2);
  double diffDelta = 1e-7;
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, diffDelta, 1e-5);
}

TEST(CPStateCostFactor, Factor) {
  Vector5 x1;
  x1 << 1, 1, M_PI / 3, 1, 1;
  CPStateCostFactor factor(example::x1_key, 
                            example::cost_model1);

  Vector4 actual_errors, expected_errors;

  actual_errors = factor.evaluateError(x1);
  expected_errors << 1, 1, 1.5, 1;

  EXPECT(assert_equal(expected_errors, actual_errors, 1e-5));
  // Make sure linearization is correct
  Values values;
  values.insert(example::x1_key, x1);
  double diffDelta = 1e-10;
  EXPECT_CORRECT_FACTOR_JACOBIANS(factor, values, diffDelta, 1e-2);
}



/* main function */
int main() {
  TestResult tr;
  return TestRegistry::runAllTests(tr);
}
