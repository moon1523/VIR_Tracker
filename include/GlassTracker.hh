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

#ifndef GlassTracker_hh
#define GlassTracker_hh

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
#include "utils.hpp"

using namespace std;

class GlassTracker
{
public:
    GlassTracker(string record_file="");
    ~GlassTracker();

    bool Run(cv::Mat &image);
    void Render();
    
    void Set_Parameters(string camParam, string detParam);
    void Set_CameraPose(const Eigen::Affine3f& a) {
      A_viewCam = a;
      A_viewCam_inverse = a.inverse();
    }
    Eigen::Affine3f Get_Glass_Pose() { return A_glass; }
    string Get_Glass_Pose_as_String() {
        Eigen::Vector3f t = A_glass.translation();
        Eigen::Quaternionf q = Eigen::Quaternionf(A_glass.rotation());
        string str = "glass\n";
        str += to_string(t.x()) + " " + to_string(t.y()) + " " + to_string(t.z()) + " " +
               to_string(q.x()) + " " + to_string(q.y()) + " " + to_string(q.z()) + " " + to_string(q.w());
        return str;
    }

    cv::VideoCapture Get_VideoCapture() { return cap; }

private:
    cv::Mat camMatrix, distCoeffs;
    cv::Ptr<cv::aruco::DetectorParameters> params;
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    float markerLength; 
    cv::Mat image, display;
    

    map<int, vector<cv::Point2f>> corner_cumul;
    cv::Vec3d tvec_cumul;
    Eigen::Quaternionf q_cumul;
    Eigen::Vector3f t_cumul;
    int cumulCount;
    vector<double> coeffX;
    vector<double> coeffY;

    Eigen::Quaternionf q_current;
    Eigen::Vector3f t_current;

    Eigen::Affine3f A_glass, A_viewCam, A_viewCam_inverse;

    cv::VideoCapture cap;
};
#endif