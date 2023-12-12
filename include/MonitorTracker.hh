/*
By downloading, copying, installing or using the software you agree to this
license. If you do not agree to this license, do not download, install,
copy or use the software.

                          License Agreement
               For Open Source Computer Vision Library
                       (3-clause BSD License)

Copyright (C) 2013, OpenCV Foundation, all rights reserved.
Third party copyrights are property of their respective owners.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  * Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.

  * Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

  * Neither the names of the copyright holders nor the names of the contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall copyright holders or contributors be liable for
any direct, indirect, incidental, special, exemplary, or consequential damages
(including, but not limited to, procurement of substitute goods or services;
loss of use, data, or profits; or business interruption) however caused
and on any theory of liability, whether in contract, strict liability,
or tort (including negligence or otherwise) arising in any way out of
the use of this software, even if advised of the possibility of such damage.
*/

#ifndef MonitorTracker_HH_
#define MonitorTracker_HH_

#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include <opencv2/highgui.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <iostream>
#include <fstream>
#include <tuple>

#include "utils.hpp"

using namespace std;

class MonitorTracker
{
  public:
    MonitorTracker(string record_file="");
    ~MonitorTracker();

    void Run(cv::Mat &image);
    void Render();
    void Render(cv::Mat& image);    
    void CollectData(int frameNo);

    void Set_Parameters(string camParam, string detParam);
    void Set_CameraPose(const Eigen::Affine3f& a) {
      A_viewCam = a;
      A_viewCam_inverse = a.inverse();
    }
    void Set_MonitorPose0(string monitor_a0) {
      ifstream ifs(monitor_a0);
      if(!ifs.is_open()) {
          cout << "'monitor_a0.txt' is not opened, take the monitor using setup toolkit" << endl;
          exit(1);
      }
      cout << "Read monitor_a0.txt" << endl;
      string line;
      while(getline(ifs, line)) {
          stringstream ss(line);
          int id;
          float x, y, z, qx, qy, qz, qw;
          ss >> id >> x >> y >> z >> qx >> qy >> qz >> qw;
          Eigen::Vector3f t(x, y, z);
          Eigen::Quaternionf q(qw, qx, qy, qz); q.normalize();
          Eigen::Affine3f a = Eigen::Translation3f(t) * q;
          a0_cumuls[id] = a;
      }
      ifs.close();
      A_monitor.setIdentity();
      cout << "ready.." << endl;
    }


    
    Eigen::Affine3f Get_Monitor_Pose() { return A_monitor; }
    string Get_Monitor_Pose_as_String() {
        Eigen::Vector3f t = A_monitor.translation();
        Eigen::Quaternionf q = Eigen::Quaternionf(A_monitor.rotation());
        string str = to_string(t.x()) + " " + to_string(t.y()) + " " + to_string(t.z()) + " " +
                     to_string(q.x()) + " " + to_string(q.y()) + " " + to_string(q.z()) + " " + to_string(q.w());
        return str;
    }
    cv::VideoCapture Get_VideoCapture() { return cap; }
    map<int, Eigen::Affine3f> Get_FrameData_Monitor_Pose() { return frameData_monitor_pose; }
    map<int, cv::Mat> Get_FrameData_Monitor_Image() { return frameData_monitor_image; }

  private:
    cv::Mat camMatrix, distCoeffs;
    cv::Ptr<cv::aruco::DetectorParameters> params;
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    float markerLength;
    cv::Mat image, display;
    int nMarkers;

    map<int, cv::Vec3f> rvec0_cumuls, rveci_cumuls;
    map<int, cv::Vec3f> tvec0_cumuls, tveci_cumuls;
    map<int, Eigen::Quaternionf> q0_cumuls, qi_cumuls;
    map<int, Eigen::Vector3f> t0_cumuls, ti_cumuls;
    map<int, Eigen::Affine3f> a0_cumuls, ai_cumuls;

    vector<Eigen::Affine3f> As;
    Eigen::Affine3f A_monitor, A_viewCam, A_viewCam_inverse;

    map<int, vector<cv::Point2f>> corner_cumul;
    int cumulCount;

    cv::VideoCapture cap;
    map<int, Eigen::Affine3f> frameData_monitor_pose;
    map<int, cv::Mat> frameData_monitor_image;
};



#endif