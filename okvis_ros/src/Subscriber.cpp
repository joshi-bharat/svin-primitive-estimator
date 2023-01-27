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
 *  Created on: Mar 23, 2012
 *      Author: Stefan Leutenegger (s.leutenegger@imperial.ac.uk)
 *    Modified: Andreas Forster (an.forster@gmail.com)
 *********************************************************************************/

/**
 * @file Subscriber.cpp
 * @brief Source file for the Subscriber class.
 * @author Stefan Leutenegger
 * @author Andreas Forster
 */

#include <glog/logging.h>

#include <functional>
#include <memory>
#include <okvis/Subscriber.hpp>
#include <vector>

#define THRESHOLD_DATA_DELAY_WARNING 0.1  // in seconds

/// \brief okvis Main namespace of this package.
namespace okvis {

Subscriber::~Subscriber() {
  if (imgTransport_ != 0) delete imgTransport_;
}

Subscriber::Subscriber(ros::NodeHandle& nh,
                       okvis::VioInterface* vioInterfacePtr,
                       const okvis::VioParametersReader& param_reader)
    : vioInterface_(vioInterfacePtr) {
  /// @Sharmin
  tfBuffer_.reset(new tf2_ros::Buffer());
  tfListener_.reset(new tf2_ros::TransformListener(*tfBuffer_));
  param_reader.getParameters(vioParameters_);
  imgTransport_ = 0;
  if (param_reader.useDriver) {
#ifdef HAVE_LIBVISENSOR
    if (param_reader.viSensor != nullptr)
      sensor_ = std::static_pointer_cast<visensor::ViSensorDriver>(param_reader.viSensor);
    initialiseDriverCallbacks();
    std::vector<unsigned int> camRate(vioParameters_.nCameraSystem.numCameras(),
                                      vioParameters_.sensors_information.cameraRate);
    startSensors(camRate, vioParameters_.imu.rate);
    // init dynamic reconfigure
    cameraConfigReconfigureService_.setCallback(boost::bind(&Subscriber::configCallback, this, _1, _2));
#else
    LOG(ERROR) << "Configuration specified to directly access the driver. "
               << "However the visensor library was not found. Trying to set up ROS nodehandle instead";
    setNodeHandle(nh);
#endif
  } else {
    setNodeHandle(nh);
  }
  imgLeftCounter = 0;   // @Sharmin
  imgRightCounter = 0;  // @Sharmin
  // Added by Sharmin
  if (vioParameters_.histogramParams.histogramMethod == HistogramMethod::CLAHE) {
    clahe = cv::createCLAHE();
    clahe->setClipLimit(vioParameters_.histogramParams.claheClipLimit);
    clahe->setTilesGridSize(
        cv::Size(vioParameters_.histogramParams.claheTilesGridSize, vioParameters_.histogramParams.claheTilesGridSize));

    std::cout << "Set Clahe Params " << vioParameters_.histogramParams.claheClipLimit << " "
              << vioParameters_.histogramParams.claheTilesGridSize << std::endl;
  }
}

void Subscriber::setNodeHandle(ros::NodeHandle& nh) {
  nh_ = &nh;

  imageSubscribers_.resize(vioParameters_.nCameraSystem.numCameras());

  // set up image reception
  if (imgTransport_ != 0) delete imgTransport_;
  imgTransport_ = new image_transport::ImageTransport(nh);

  // set up callbacks
  for (size_t i = 0; i < vioParameters_.nCameraSystem.numCameras(); ++i) {
    imageSubscribers_[i] =
        imgTransport_->subscribe("/camera" + std::to_string(i),
                                 100 * vioParameters_.nCameraSystem.numCameras(),
                                 std::bind(&Subscriber::imageCallback, this, std::placeholders::_1, i));
  }

  subImu_ = nh_->subscribe("/imu", 1000, &Subscriber::imuCallback, this);

  // Sharmin
  if (vioParameters_.sensorList.isSonarUsed) {
    subSonarRange_ = nh_->subscribe("/imagenex831l/range", 1000, &Subscriber::sonarCallback, this);
  }
  // Sharmin
  // if (vioParameters_.sensorList.isDepthUsed){
  // subDepth_ = nh_->subscribe("/bar30/depth", 1000, &Subscriber::depthCallback, this);
  // subDepth_ = nh_->subscribe("/aqua/state", 1000, &Subscriber::depthCallback, this); // Aqua depth topic
  // }

  // Sharmin
  if (vioParameters_.relocParameters.isRelocalization) {
    std::cout << "Subscribing to /pose_graph/match_points topic" << std::endl;
    subReloPoints_ = nh_->subscribe("/pose_graph/match_points", 1000, &Subscriber::relocCallback, this);
  }
}

// Hunter
void Subscriber::setT_Wc_W(okvis::kinematics::Transformation T_Wc_W) { vioParameters_.publishing.T_Wc_W = T_Wc_W; }

void Subscriber::imageCallback(const sensor_msgs::ImageConstPtr& msg, unsigned int cameraIndex) {
  const cv::Mat raw = readRosImage(msg);

  // resizing factor( e.g., with a factor = 0.8, an image will convert from 800x600 to 640x480)
  cv::Mat raw_resized;
  if (vioParameters_.miscParams.resizeFactor != 1.0) {
    cv::resize(
        raw, raw_resized, cv::Size(), vioParameters_.miscParams.resizeFactor, vioParameters_.miscParams.resizeFactor);
  } else {
    raw_resized = raw.clone();
  }

  cv::Mat filtered;
  if (vioParameters_.optimization.useMedianFilter) {
    cv::medianBlur(raw_resized, filtered, 3);
  } else {
    filtered = raw_resized.clone();
  }

  // Added by Sharmin for CLAHE
  cv::Mat histogram_equalized_image;
  if (vioParameters_.histogramParams.histogramMethod == HistogramMethod::CLAHE) {
    clahe->apply(filtered, histogram_equalized_image);
  } else if (vioParameters_.histogramParams.histogramMethod == HistogramMethod::HISTOGRAM) {
    cv::equalizeHist(filtered, histogram_equalized_image);
  } else {
    histogram_equalized_image = filtered;
  }
  // End Added by Sharmin

  // adapt timestamp
  okvis::Time t(msg->header.stamp.sec, msg->header.stamp.nsec);
  t -= okvis::Duration(vioParameters_.sensors_information.imageDelay);

  if (!vioInterface_->addImage(t, cameraIndex, histogram_equalized_image)) {
    // LOG(WARNING) << "Frame delayed at time " << t;
  }
}

void Subscriber::imuCallback(const sensor_msgs::ImuConstPtr& msg) {
  vioInterface_->addImuMeasurement(
      okvis::Time(msg->header.stamp.sec, msg->header.stamp.nsec),
      Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z),
      Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z));
}

// @Sharmin
void Subscriber::relocCallback(const sensor_msgs::PointCloudConstPtr& relo_msg) {
  std::vector<Eigen::Vector3d> matched_ids;
  okvis::kinematics::Transformation T_Wc_W = vioParameters_.publishing.T_Wc_W;
  // double frame_stamp = relo_msg->header.stamp.toSec();
  for (unsigned int i = 0; i < relo_msg->points.size(); i++) {
    // landmarkId, mfId/poseId, keypointIdx for Every Matched 3d points in Current frame
    Eigen::Vector4d pt_ids;
    pt_ids.x() = relo_msg->points[i].x;
    pt_ids.y() = relo_msg->points[i].y;
    pt_ids.z() = relo_msg->points[i].z;
    pt_ids.w() = 1.0;
    pt_ids = T_Wc_W.inverse() * pt_ids;  // Hunter: Transform reloc points
    matched_ids.push_back(pt_ids.segment<3>(0));
  }
  Eigen::Vector3d pos(
      relo_msg->channels[0].values[0], relo_msg->channels[0].values[1], relo_msg->channels[0].values[2]);
  Eigen::Quaterniond quat(relo_msg->channels[0].values[3],
                          relo_msg->channels[0].values[4],
                          relo_msg->channels[0].values[5],
                          relo_msg->channels[0].values[6]);
  // Eigen::Matrix3d relo_r = relo_q.toRotationMatrix();

  // Hunter: Transform reloc pose
  okvis::kinematics::Transformation pose_Wc(pos, quat);
  okvis::kinematics::Transformation pose_W = T_Wc_W.inverse() * pose_Wc;

  // estimator.setReloFrame(frame_stamp, frame_index, match_points, relo_t, relo_r);

  vioInterface_->addRelocMeasurement(
      okvis::Time(relo_msg->header.stamp.sec, relo_msg->header.stamp.nsec), matched_ids, pose_W.r(), pose_W.q());
}
// @Sharmin
// /*
// // Aqua depth topic subscription
// void Subscriber::depthCallback(const aquacore::StateMsg::ConstPtr& msg)
// {
//         vioInterface_->addDepthMeasurement(
//                                           okvis::Time(msg->header.stamp.sec, msg->header.stamp.nsec),
//                                           msg->Depth);
// }
// */

// /*
// // stereo rig depth topic subscription
// void Subscriber::depthCallback(const depth_node_py::Depth::ConstPtr& msg)
// {
//         vioInterface_->addDepthMeasurement(
//                                           okvis::Time(msg->header.stamp.sec, msg->header.stamp.nsec),
//                                           msg->depth);
// }
// * /

// @Sharmin
void Subscriber::sonarCallback(const imagenex831l::ProcessedRange::ConstPtr& msg) {
  double rangeResolution = msg->max_range / msg->intensity.size();
  int max = 0;
  int maxIndex = 0;

  // @Sharmin: discarding as range was set higher (which introduced some noisy data) during data collection
  for (unsigned int i = 0; i < msg->intensity.size() - 100; i++) {
    if (msg->intensity[i] > max) {
      max = msg->intensity[i];
      maxIndex = i;
    }
  }

  double range = (maxIndex + 1) * rangeResolution;
  double heading = (msg->head_position * M_PI) / 180;

  // No magic no!! within 4.5 meter
  if (range < 4.5 && max > 10) {
    vioInterface_->addSonarMeasurement(okvis::Time(msg->header.stamp.sec, msg->header.stamp.nsec), range, heading);
  }
}

const cv::Mat Subscriber::readRosImage(const sensor_msgs::ImageConstPtr& img_msg) const {
  CHECK(img_msg);
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    // TODO(Toni): here we should consider using toCvShare...
    cv_ptr = cv_bridge::toCvCopy(img_msg);
  } catch (cv_bridge::Exception& exception) {
    ROS_FATAL("cv_bridge exception: %s", exception.what());
    ros::shutdown();
  }

  CHECK(cv_ptr);
  const cv::Mat img_const = cv_ptr->image;  // Don't modify shared image in ROS.
  cv::Mat converted_img(img_const.size(), CV_8U);
  if (img_msg->encoding == sensor_msgs::image_encodings::BGR8) {
    // LOG_EVERY_N(WARNING, 10) << "Converting image...";
    cv::cvtColor(img_const, converted_img, cv::COLOR_BGR2GRAY);
    return converted_img;
  } else if (img_msg->encoding == sensor_msgs::image_encodings::RGB8) {
    // LOG_EVERY_N(WARNING, 10) << "Converting image...";
    cv::cvtColor(img_const, converted_img, cv::COLOR_RGB2GRAY);
    return converted_img;
  } else {
    CHECK_EQ(cv_ptr->encoding, sensor_msgs::image_encodings::MONO8)
        << "Expected image with MONO8, BGR8, or RGB8 encoding."
           "Add in here more conversions if you wish.";
    return img_const;
  }
}

#ifdef HAVE_LIBVISENSOR
void Subscriber::initialiseDriverCallbacks() {
  // mostly copied from https://github.com/ethz-asl/visensor_node_devel
  if (sensor_ == nullptr) {
    sensor_ = std::unique_ptr<visensor::ViSensorDriver>(new visensor::ViSensorDriver());

    try {
      // use autodiscovery to find sensor. TODO: specify IP in config?
      sensor_->init();
    } catch (Exception const& ex) {
      LOG(ERROR) << ex.what();
      exit(1);
    }
  }

  try {
    sensor_->setCameraCallback(
        std::bind(&Subscriber::directFrameCallback, this, std::placeholders::_1, std::placeholders::_2));
    sensor_->setImuCallback(
        std::bind(&Subscriber::directImuCallback, this, std::placeholders::_1, std::placeholders::_2));
    sensor_->setFramesCornersCallback(
        std::bind(&Subscriber::directFrameCornerCallback, this, std::placeholders::_1, std::placeholders::_2));
    sensor_->setCameraCalibrationSlot(0);  // 0 is factory calibration
  } catch (Exception const& ex) {
    LOG(ERROR) << ex.what();
  }
}
#endif

#ifdef HAVE_LIBVISENSOR
void Subscriber::startSensors(const std::vector<unsigned int>& camRate, const unsigned int imuRate) {
  // mostly copied from https://github.com/ethz-asl/visensor_node_devel
  OKVIS_ASSERT_TRUE_DBG(Exception, sensor_ != nullptr, "Sensor pointer not yet initialised.");

  std::vector<visensor::SensorId::SensorId> listOfCameraIds = sensor_->getListOfCameraIDs();

  OKVIS_ASSERT_TRUE_DBG(Exception, listOfCameraIds.size() == camRate.size(), "Number of cameras don't match up.");

  for (uint i = 0; i < listOfCameraIds.size(); i++) {
    if (camRate[i] > 0) sensor_->startSensor(listOfCameraIds[i], camRate[i]);
  }

  sensor_->startAllCorners();
  sensor_->startSensor(visensor::SensorId::IMU0, imuRate);
  // /*if (sensor_->isSensorPresent(visensor::SensorId::LED_FLASHER0))
  // sensor_->startSensor(visensor::SensorId::LED_FLASHER0);*/ // apparently experimental...
}
#endif

#ifdef HAVE_LIBVISENSOR
void Subscriber::directImuCallback(boost::shared_ptr<visensor::ViImuMsg> imu_ptr, visensor::ViErrorCode error) {
  if (error == visensor::ViErrorCodes::MEASUREMENT_DROPPED) {
    LOG(WARNING) << "dropped imu measurement on sensor " << imu_ptr->imu_id << " (check network bandwidth/sensor rate)";
    return;
  }

  okvis::Time timestamp;
  timestamp.fromNSec(imu_ptr->timestamp);

  vioInterface_->addImuMeasurement(timestamp,
                                   Eigen::Vector3d(imu_ptr->acc[0], imu_ptr->acc[1], imu_ptr->acc[2]),
                                   Eigen::Vector3d(imu_ptr->gyro[0], imu_ptr->gyro[1], imu_ptr->gyro[2]));
}
#endif

#ifdef HAVE_LIBVISENSOR
void Subscriber::directFrameCallback(visensor::ViFrame::Ptr frame_ptr, visensor::ViErrorCode error) {
  if (error == visensor::ViErrorCodes::MEASUREMENT_DROPPED) {
    LOG(WARNING) << "dropped camera image on sensor " << frame_ptr->camera_id
                 << " (check network bandwidth/sensor rate)";
    return;
  }

  int image_height = frame_ptr->height;
  int image_width = frame_ptr->width;

  okvis::Time timestamp;
  timestamp.fromNSec(frame_ptr->timestamp);

  // check if transmission is delayed
  const double frame_delay = (okvis::Time::now() - timestamp).toSec();
  if (frame_delay > THRESHOLD_DATA_DELAY_WARNING)
    LOG(WARNING) << "Data arrived later than expected [ms]: " << frame_delay * 1000.0;

  cv::Mat raw;
  if (frame_ptr->image_type == visensor::MONO8) {
    raw = cv::Mat(image_height, image_width, CV_8UC1);
    memcpy(raw.data, frame_ptr->getImageRawPtr(), image_width * image_height);
  } else if (frame_ptr->image_type == visensor::MONO16) {
    raw = cv::Mat(image_height, image_width, CV_16UC1);
    memcpy(raw.data, frame_ptr->getImageRawPtr(), (image_width)*image_height * 2);
  } else {
    LOG(WARNING) << "[VI_SENSOR] - unknown image type!";
    return;
  }

  cv::Mat filtered;
  if (vioParameters_.optimization.useMedianFilter) {
    cv::medianBlur(raw, filtered, 3);
  } else {
    filtered = raw.clone();
  }

  // adapt timestamp
  timestamp -= okvis::Duration(vioParameters_.sensors_information.imageDelay);

  if (!vioInterface_->addImage(timestamp, frame_ptr->camera_id, filtered))
    LOG(WARNING) << "Frame delayed at time " << timestamp;
}
#endif

#ifdef HAVE_LIBVISENSOR
void Subscriber::directFrameCornerCallback(visensor::ViFrame::Ptr /*frame_ptr*/,
                                           visensor::ViCorner::Ptr /*corners_ptr*/) {
  LOG(INFO) << "directframecornercallback";
}
#endif

#ifdef HAVE_LIBVISENSOR
void Subscriber::configCallback(okvis_ros::CameraConfig& config, uint32_t level) {
  if (sensor_ == nullptr) {
    return;  // not yet set up -- do nothing...
  }

  std::vector<visensor::SensorId::SensorId> listOfCameraIds = sensor_->getListOfCameraIDs();

  // adopted from visensor_node, see https://github.com/ethz-asl/visensor_node.git
  // configure MPU 9150 IMU (if available)
  if (std::count(listOfCameraIds.begin(), listOfCameraIds.end(), visensor::SensorId::IMU_CAM0) > 0)
    sensor_->setSensorConfigParam(visensor::SensorId::IMU_CAM0, "digital_low_pass_filter_config", 0);

  if (std::count(listOfCameraIds.begin(), listOfCameraIds.end(), visensor::SensorId::IMU_CAM1) > 0)
    sensor_->setSensorConfigParam(visensor::SensorId::IMU_CAM1, "digital_low_pass_filter_config", 0);

  // ========================= CAMERA 0 ==========================
  if (std::count(listOfCameraIds.begin(), listOfCameraIds.end(), visensor::SensorId::CAM0) > 0) {
    sensor_->setSensorConfigParam(visensor::SensorId::CAM0, "agc_enable", config.cam0_agc_enable);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM0, "max_analog_gain", config.cam0_max_analog_gain);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM0, "global_analog_gain", config.cam0_global_analog_gain);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM0, "global_analog_gain_attenuation", config.cam0_global_analog_gain_attenuation);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM0, "aec_enable", config.cam0_aec_enable);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM0, "min_coarse_shutter_width", config.cam0_min_coarse_shutter_width);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM0, "max_coarse_shutter_width", config.cam0_max_coarse_shutter_width);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM0, "coarse_shutter_width", config.cam0_coarse_shutter_width);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM0, "fine_shutter_width", config.cam0_fine_shutter_width);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM0, "adc_mode", config.cam0_adc_mode);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM0, "vref_adc_voltage_level", config.cam0_vref_adc_voltage_level);
  }

  // ========================= CAMERA 1 ==========================
  if (std::count(listOfCameraIds.begin(), listOfCameraIds.end(), visensor::SensorId::CAM1) > 0) {
    sensor_->setSensorConfigParam(visensor::SensorId::CAM1, "agc_enable", config.cam1_agc_enable);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM1, "max_analog_gain", config.cam1_max_analog_gain);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM1, "global_analog_gain", config.cam1_global_analog_gain);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM1, "global_analog_gain_attenuation", config.cam1_global_analog_gain_attenuation);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM1, "aec_enable", config.cam1_aec_enable);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM1, "min_coarse_shutter_width", config.cam1_min_coarse_shutter_width);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM1, "max_coarse_shutter_width", config.cam1_max_coarse_shutter_width);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM1, "coarse_shutter_width", config.cam1_coarse_shutter_width);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM1, "adc_mode", config.cam1_adc_mode);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM1, "vref_adc_voltage_level", config.cam1_vref_adc_voltage_level);
  }

  // ========================= CAMERA 2 ==========================
  if (std::count(listOfCameraIds.begin(), listOfCameraIds.end(), visensor::SensorId::CAM2) > 0) {
    sensor_->setSensorConfigParam(visensor::SensorId::CAM2, "agc_enable", config.cam2_agc_enable);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM2, "max_analog_gain", config.cam2_max_analog_gain);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM2, "global_analog_gain", config.cam2_global_analog_gain);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM2, "global_analog_gain_attenuation", config.cam2_global_analog_gain_attenuation);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM2, "aec_enable", config.cam2_aec_enable);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM2, "min_coarse_shutter_width", config.cam2_min_coarse_shutter_width);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM2, "max_coarse_shutter_width", config.cam2_max_coarse_shutter_width);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM2, "coarse_shutter_width", config.cam2_coarse_shutter_width);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM2, "adc_mode", config.cam2_adc_mode);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM2, "vref_adc_voltage_level", config.cam2_vref_adc_voltage_level);
  }

  // ========================= CAMERA 3 ==========================
  if (std::count(listOfCameraIds.begin(), listOfCameraIds.end(), visensor::SensorId::CAM3) > 0) {
    sensor_->setSensorConfigParam(visensor::SensorId::CAM3, "agc_enable", config.cam3_agc_enable);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM3, "max_analog_gain", config.cam3_max_analog_gain);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM3, "global_analog_gain", config.cam3_global_analog_gain);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM3, "global_analog_gain_attenuation", config.cam3_global_analog_gain_attenuation);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM3, "aec_enable", config.cam3_aec_enable);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM3, "min_coarse_shutter_width", config.cam3_min_coarse_shutter_width);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM3, "max_coarse_shutter_width", config.cam3_max_coarse_shutter_width);
    sensor_->setSensorConfigParam(visensor::SensorId::CAM3, "coarse_shutter_width", config.cam3_coarse_shutter_width);

    sensor_->setSensorConfigParam(visensor::SensorId::CAM3, "adc_mode", config.cam3_adc_mode);
    sensor_->setSensorConfigParam(
        visensor::SensorId::CAM3, "vref_adc_voltage_level", config.cam3_vref_adc_voltage_level);
  }
}
#endif

}  // namespace okvis
