#include "GlassTracker.hh"

GlassTracker::GlassTracker(string record_file)
: cumulCount(0)
{
    dictionary = cv::aruco::generateCustomDictionary(6, 4, 2);
    params = cv::aruco::DetectorParameters::create();
    markerLength = 8; // cm
    // marker position (relative position-based methodology)
    coeffX = {25, 25, 18, -25, -18}; // cm
    coeffY = {-27, 0, 33, 0, -33};   // cm

    if (!record_file.empty()) {
        cap.open(record_file);
    }
    else {
        cap.open(findCameraIndex(), cv::CAP_V4L2);
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH,  2560);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 1440);
        cap.set(cv::CAP_PROP_FPS, 15);
        cap.set(cv::CAP_PROP_AUTOFOCUS, 0);
    }
    
    if (!cap.isOpened()) {
        cerr << "Error! Unabale to open camera" << endl;
        exit(1);
    }
}

GlassTracker::~GlassTracker()
{
    
}

void GlassTracker::Set_Parameters(string camParam, string detParam)
{
    cv::FileStorage fs(camParam, cv::FileStorage::READ);
    if (!fs.isOpened())
        return;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    cv::FileStorage fs2(detParam, cv::FileStorage::READ);
    if (!fs2.isOpened())
        return;
    fs2["adaptiveThreshWinSizeMin"] >> params->adaptiveThreshWinSizeMin;
    fs2["adaptiveThreshWinSizeMax"] >> params->adaptiveThreshWinSizeMax;
    fs2["adaptiveThreshWinSizeStep"] >> params->adaptiveThreshWinSizeStep;
    fs2["adaptiveThreshConstant"] >> params->adaptiveThreshConstant;
    fs2["minMarkerPerimeterRate"] >> params->minMarkerPerimeterRate;
    fs2["maxMarkerPerimeterRate"] >> params->maxMarkerPerimeterRate;
    fs2["polygonalApproxAccuracyRate"] >> params->polygonalApproxAccuracyRate;
    fs2["minCornerDistanceRate"] >> params->minCornerDistanceRate;
    fs2["minDistanceToBorder"] >> params->minDistanceToBorder;
    fs2["minMarkerDistanceRate"] >> params->minMarkerDistanceRate;
    fs2["cornerRefinementMethod"] >> params->cornerRefinementMethod;
    fs2["cornerRefinementWinSize"] >> params->cornerRefinementWinSize;
    fs2["cornerRefinementMaxIterations"] >> params->cornerRefinementMaxIterations;
    fs2["cornerRefinementMinAccuracy"] >> params->cornerRefinementMinAccuracy;
    fs2["markerBorderBits"] >> params->markerBorderBits;
    fs2["perspectiveRemovePixelPerCell"] >> params->perspectiveRemovePixelPerCell;
    fs2["perspectiveRemoveIgnoredMarginPerCell"] >> params->perspectiveRemoveIgnoredMarginPerCell;
    fs2["maxErroneousBitsInBorderRate"] >> params->maxErroneousBitsInBorderRate;
    fs2["minOtsuStdDev"] >> params->minOtsuStdDev;
    fs2["errorCorrectionRate"] >> params->errorCorrectionRate;
}

bool GlassTracker::Run(cv::Mat &image)
{
    image.copyTo(display);   

    // detect markers
    vector<int> ids;
    vector<vector<cv::Point2f>> corners, rejected, cornersWhole;
    cv::aruco::detectMarkers(image, dictionary, corners, ids, params, rejected);

    // compare with the previous result
    if (ids.size() > 0)
    {
        bool isNewPose(false);
        for (int i = 0; i < ids.size(); i++)
        {
            vector<cv::Point2f> points;
            for (auto p : corners[i])
                points.push_back( cv::Point2f(p.x, p.y) );
            cornersWhole.push_back(points);
            if (isNewPose)
                continue;
            if (corner_cumul.find(ids[i]) != corner_cumul.end())
            {
                cv::Point2f oldCen(0, 0), newCen(0, 0);
                for (int n = 0; n < 4; n++)
                {
                    newCen += points[n];
                    oldCen += corner_cumul[ids[i]][n];
                }
                cv::Point2f vec = oldCen - newCen;
                if (vec.dot(vec) > 40)
                {
                    corner_cumul.clear();
                    isNewPose = true;
                    cumulCount = 0;
                }
                else
                    cumulCount++;
            }
        }
        for (int i = 0; i < ids.size(); i++)
            corner_cumul[ids[i]] = cornersWhole[i];

        vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::estimatePoseSingleMarkers(cornersWhole, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

        //rvec
        std::vector<Eigen::Vector4f> q_vec;
        for (cv::Vec3d v : rvecs)
        {
            double angle = norm(v);
            Eigen::Vector3d axis(v(0) / angle, v(1) / angle, v(2) / angle);
            Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));
            q.normalize();
            q_vec.push_back(Eigen::Vector4f(q.x(), q.y(), q.z(), q.w()));
        }
        Eigen::Quaternionf q_avg(averaging_quaternions(q_vec));
        if (cumulCount > 1)
            q_cumul = q_cumul.slerp(1.f / (cumulCount + 1.f), q_avg);
        else
            q_cumul = q_avg;

        Eigen::AngleAxisf avg(q_cumul);
        cv::Vec3f rvec;
        cv::eigen2cv(avg.axis(), rvec);
        rvec *= avg.angle();
        cv::Vec3f axisX, axisY, axisZ;
        cv::eigen2cv(q_cumul * Eigen::Vector3f(1.f, 0.f, 0.f), axisX);
        cv::eigen2cv(q_cumul * Eigen::Vector3f(0.f, 1.f, 0.f), axisY);
        cv::eigen2cv(q_cumul * Eigen::Vector3f(0.f, 0.f, 1.f), axisZ);

        //tvec
        cv::Vec3d tvec(0, 0, 0);
        for (int i = 0; i < ids.size(); i++)
        {
            int id = ids[i];
            cv::Vec3d xTrans = coeffX[id] * axisX;
            cv::Vec3d yTrans = coeffY[id] * axisY;
            tvec += (tvecs[i] + xTrans + yTrans);
        }
        tvec *= 1.f / ids.size();
        if (cumulCount > 1)
        {
            tvec += tvec_cumul * (cumulCount - 1);
            tvec /= (double)cumulCount;
        }
        tvec_cumul = tvec;
        cv::aruco::drawAxis(display, camMatrix, distCoeffs, rvec, tvec, markerLength * 3.f);

        q_current = q_cumul;
        cv::cv2eigen(cv::Vec3f(tvec_cumul), t_current);

        A_glass.linear() = q_current.normalized().toRotationMatrix();
        A_glass.translation() = t_current * 0.01; // cm to meter
        // A_glass = A_viewCam * A_glass;
    }
    else
    {
        corner_cumul.clear();
        cumulCount = 0;
        return false;
    }

    //draw result
    if (corner_cumul.size() > 0)
    {
        vector<vector<cv::Point2f>> cornersTemp;
        vector<int> idsTemp;
        for (auto iter : corner_cumul)
        {
            cornersTemp.push_back(iter.second);
            idsTemp.push_back(iter.first);
        }
        cv::aruco::drawDetectedMarkers(display, cornersTemp, idsTemp);
    }
    return true;
}

void GlassTracker::Render()
{
    cv::resize(display, display, cv::Size(1280, 720));
    cv::imshow("Render", display);
}
