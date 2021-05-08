/* ----------------------------------------------------------------------------
 * GTDynamics Copyright 2020, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

/**
 * @file  CartpoleFactors.h
 * @brief Cart pole dynamcis and cost factors
 * @Author: Yetong Zhang
 */


#pragma once

#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

#include <boost/optional.hpp>
#include <iostream>
#include <string>

namespace gtdynamics {

/** CPDynamicsFactor */
class CPDynamicsFactor : public gtsam::NoiseModelFactor2<gtsam::Vector5, gtsam::Vector5> {
 private:
  typedef CPDynamicsFactor This;
  typedef gtsam::NoiseModelFactor2<gtsam::Vector5, gtsam::Vector5> Base;
  double dt_;

 public:
  CPDynamicsFactor(gtsam::Key x1_key, gtsam::Key x2_key,
               const gtsam::noiseModel::Base::shared_ptr &cost_model, double dt)
      : Base(cost_model, x1_key, x2_key), dt_(dt) {}
  virtual ~CPDynamicsFactor() {}

 private:
 public:
  gtsam::Vector evaluateError(
      const gtsam::Vector5 &x1, const gtsam::Vector5 &x2,
      boost::optional<gtsam::Matrix &> H_x1 = boost::none,
      boost::optional<gtsam::Matrix &> H_x2 = boost::none) const override {

    double x = x1(0);
    double v = x1(1);
    double theta = x1(2);
    double omega = x1(3);
    double f = x1(4);

    double sint = sin(theta);
    double cost = cos(theta);

    double mc = 1;
    double mp = 0.01;
    double g = 9.81;
    double l = 0.25;
    double mass_eq = mc + mp * pow(sint, 2);
    double moment_eq = l * mass_eq;

    double dx = v * dt_;
    double dv_f_eq = f + mp*sint * (l*pow(omega,2) + g*cost);
    double dv = dv_f_eq / mass_eq * dt_;
    double dtheta = omega * dt_;
    double domega_f_eq = -f * cost - mp*l*pow(omega,2)*cost*sint - (mc+mp)*g*sint;
    double domega = domega_f_eq / moment_eq * dt_;
    double df = -20 * f * dt_;


    if (H_x1) {
      double term_dv_1 = -2*mp*sint*cost / pow(mass_eq, 2);
      double term_dv_2 = mp * cost * (l*pow(omega,2)+g*cost) - mp*sint*g*sint;
      double term_domega_1 = term_dv_1/l;
      double term_domega_2 = f * sint - (mc+mp)*g*cost - mp*l*pow(omega,2)*(-sint*sint + cost*cost);

      *H_x1 = gtsam::I_5x5;
      (*H_x1)(0,1) = dt_;
      (*H_x1)(1,2) = dt_ * (dv_f_eq * term_dv_1 + term_dv_2 / mass_eq);
      (*H_x1)(1,3) = dt_ * mp * sint * l / mass_eq * 2 * omega;
      (*H_x1)(1,4) = dt_ / mass_eq;
      (*H_x1)(2,3) = dt_;
      (*H_x1)(3,2) = dt_ * (domega_f_eq * term_domega_1 + term_domega_2 / moment_eq);
      (*H_x1)(3,3) = 1-dt_ * mp * l * cost * sint * 2 * omega / moment_eq;
      (*H_x1)(3,4) = -dt_ * cost / moment_eq;
      (*H_x1)(4,4) = 1-20*dt_;
    }
    if (H_x2) {
      *H_x2 = -gtsam::I_5x5;
    }

    gtsam::Vector5 error;
    error << x+dx-x2(0), v+dv-x2(1), theta+dtheta-x2(2), omega+domega-x2(3), f+df-x2(4);
    return error;
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string &s = "",
             const gtsam::KeyFormatter &keyFormatter =
                 gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "CP Dynamics factor" << std::endl;
    Base::print("", keyFormatter);
  }

 private:
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE &ar, const unsigned int version) {
    ar &boost::serialization::make_nvp(
        "NoiseModelFactor2", boost::serialization::base_object<Base>(*this));
  }
};


/** CPStateCostFactor */
class CPStateCostFactor : public gtsam::NoiseModelFactor1<gtsam::Vector5> {
 private:
  typedef CPStateCostFactor This;
  typedef gtsam::NoiseModelFactor1<gtsam::Vector5> Base;

 public:
  CPStateCostFactor(gtsam::Key x_key,
               const gtsam::noiseModel::Base::shared_ptr &cost_model)
      : Base(cost_model, x_key) {}
  virtual ~CPStateCostFactor() {}

 private:
 public:
  gtsam::Vector evaluateError(
      const gtsam::Vector5 &x,
      boost::optional<gtsam::Matrix &> H_x = boost::none) const override {
    double theta = x(2);
    double err_theta = 1 + cos(theta);
    if (H_x) {
      H_x->setConstant(4, 5, 0);
      (*H_x)(0,0) = 1;
      (*H_x)(1,1) = 1;
      (*H_x)(2,2) = -sin(theta);
      (*H_x)(3,3) = 1;
    }
    return gtsam::Vector4(x(0), x(1), err_theta, x(3));
  }

  gtsam::NonlinearFactor::shared_ptr clone() const override {
    return boost::static_pointer_cast<gtsam::NonlinearFactor>(
        gtsam::NonlinearFactor::shared_ptr(new This(*this)));
  }

  void print(const std::string &s = "",
             const gtsam::KeyFormatter &keyFormatter =
                 gtsam::DefaultKeyFormatter) const override {
    std::cout << s << "CP State Cost factor" << std::endl;
    Base::print("", keyFormatter);
  }

 private:
  friend class boost::serialization::access;
  template <class ARCHIVE>
  void serialize(ARCHIVE &ar, const unsigned int version) {
    ar &boost::serialization::make_nvp(
        "NoiseModelFactor2", boost::serialization::base_object<Base>(*this));
  }
};


}  // namespace gtdynamics