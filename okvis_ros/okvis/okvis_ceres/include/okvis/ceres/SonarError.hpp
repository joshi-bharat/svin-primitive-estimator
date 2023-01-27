/*********************************************************************************
 *  OKVIS - Open Keyframe-based Visual-Inertial SLAM
 *  Copyright (c) 2015, Autonomous Systems Lab / ETH Zurich
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *   * Neither the name of Autonomous Systems Lab / ETH Zurich nor the names of
 *     its contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 *  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Created on: Sep 10, 2013
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *********************************************************************************/

/**
 * @file SonarError.hpp
 * @brief Header file for the SonarError class.
 * @author Sharmin Rahman
 */

#ifndef INCLUDE_OKVIS_CERES_SONARERROR_HPP_
#define INCLUDE_OKVIS_CERES_SONARERROR_HPP_

#include <okvis/VioParametersReader.hpp>
#include <okvis/assert_macros.hpp>
#include <okvis/ceres/ErrorInterface.hpp>
#include <okvis/kinematics/Transformation.hpp>
#include <string>
#include <vector>

#include "ceres/ceres.h"
/// \brief okvis Main namespace of this package.
namespace okvis {
/// \brief ceres Namespace for ceres-related functionality implemented in okvis.
namespace ceres {

/// \brief Absolute error of a sonar homogeneous point (landmark).
class SonarError : public ::ceres::SizedCostFunction<1 /* number of residuals */, 7 /* size of first parameter */>,
                   public ErrorInterface {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  OKVIS_DEFINE_EXCEPTION(Exception, std::runtime_error)

  /// \brief The base class type.
  typedef ::ceres::SizedCostFunction<1, 7> base_t;

  /// \brief Number of residuals (3)
  static const int kNumResiduals = 1;

  /// \brief The information matrix type (3x3).
  typedef double information_t;

  /// \brief The covariance matrix type (same as information).
  typedef double covariance_t;

  /// \brief Default constructor.
  SonarError();

  /// \brief Construct with measurement and information matrix.
  /// @param[in] measurement The measurement.
  /// @param[in] information The information (weight) matrix.
  /* SonarError(const Eigen::Vector4d & measurement,
                         const information_t & information);*/

  /// \brief Construct with homogeneous measurement and variance.
  /// @param[in] measurement The measurement.
  /// @param[in] variance The variance of the measurement, i.e. information_ has variance in its diagonal.
  /// TODO document.
  SonarError(const okvis::VioParameters& params,
             double range,
             double heading,
             const information_t& information,
             std::vector<Eigen::Vector3d> landmarkSubset);

  /// \brief Trivial destructor.
  virtual ~SonarError() {}

  // setters
  /// \brief Set the measurement.
  /// @param[in] measurement The measurement.
  void setMeasurement(double range, double heading) {
    range_ = range;
    heading_ = heading;
  }

  /// \brief Set the landmarkSubset.
  /// @param[in] landmarkSubset The landmarkSubset.
  void setLandmarkSubset(const std::vector<Eigen::Vector3d>& landmarkSubset) { landmarkSubset_ = landmarkSubset; }

  /// \brief Set the information.
  /// @param[in] information The information (weight) matrix.
  void setInformation(const information_t& information);

  // calculates covariance
  // void computeCovarianceMatrix(const std::vector<Eigen::Vector4d> & landmarkSubset);

  // getters
  /// \brief Get the measurement.
  /// \return The measurement vector.
  double range() const { return range_; }

  double heading() const { return heading_; }

  /// \brief Get the information matrix.
  /// \return The information (weight) matrix.
  const information_t& information() const { return information_; }

  /// \brief Get the covariance matrix.
  /// \return The inverse information (covariance) matrix.
  const information_t& covariance() const { return covariance_; }

  /**
   * @brief This evaluates the error term and additionally computes the Jacobians.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @return success of th evaluation.
   */
  virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const;

  /**
   * @brief EvaluateWithMinimalJacobians This evaluates the error term and additionally computes
   *        the Jacobians in the minimal internal representation.
   * @param parameters Pointer to the parameters (see ceres)
   * @param residuals Pointer to the residual vector (see ceres)
   * @param jacobians Pointer to the Jacobians (see ceres)
   * @param jacobiansMinimal Pointer to the minimal Jacobians (equivalent to jacobians).
   * @return Success of the evaluation.
   */
  virtual bool EvaluateWithMinimalJacobians(double const* const* parameters,
                                            double* residuals,
                                            double** jacobians,
                                            double** jacobiansMinimal) const;

  // sizes
  /// \brief Residual dimension.
  size_t residualDim() const { return kNumResiduals; }

  /// \brief Number of parameter blocks.
  size_t parameterBlocks() const { return base_t::parameter_block_sizes().size(); }

  /// \brief Dimension of an individual parameter block.
  /// @param[in] parameterBlockId ID of the parameter block of interest.
  /// \return The dimension.
  size_t parameterBlockDim(size_t parameterBlockId) const {
    return base_t::parameter_block_sizes().at(parameterBlockId);
  }

  /// @brief Return parameter block type as string
  virtual std::string typeInfo() const { return "SonarError"; }

 protected:
  // the measurement
  double range_;    ///< The range measurement.
  double heading_;  ///< The heading measurement.
  std::vector<Eigen::Vector3d> landmarkSubset_;

  // weighting related
  information_t information_;            ///< The 6x6 information matrix.
  information_t _squareRootInformation;  ///< The 6x6 square root information matrix.
  covariance_t covariance_;              ///< The 6x6 covariance matrix.

 private:
  const okvis::VioParameters params_;  // Sharmin
};

}  // namespace ceres
}  // namespace okvis

#endif /* INCLUDE_OKVIS_CERES_SONARERROR_HPP_ */
