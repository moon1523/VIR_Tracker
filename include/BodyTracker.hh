#ifndef BodyTracker_HH_
#define BodyTracker_HH_

#include <sl/Camera.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Geometry>

using namespace std;

class BodyTracker
{
public:
    BodyTracker(string sn);
    ~BodyTracker();

    bool Open();
    void Run();

    void Set_TopView(string sn) { topView = stoi(sn); }
    void Set_CornerView(string sn) { cornerView = stoi(sn); }
    void Set_Monitor_TopView(string sn) { monitor_topView = stoi(sn); }
    void Set_Monitor_BotView(string sn) { monitor_botView = stoi(sn); }
    void Set_CameraPose(const Eigen::Affine3f& pose) { cam_pose = pose; }

    string Get_Pose_As_String() {
        string str;
        if (zed_sn == topView) {
            for (auto &head : objects.object_list) {
                str += "head_id " + to_string(head.id) + "\n";
                str += to_string(head.head_position.x) + " " + to_string(head.head_position.y) + " " + to_string(head.head_position.z) + "\n";
            }
        }
        for (auto &body : bodies.body_list) {
            str += "body_id " + to_string(body.id) + "\n";
            for (auto &kpt : body.keypoint) {
                str += to_string(kpt.x) + " " + to_string(kpt.y) + " " + to_string(kpt.z) + "\n";
            }
        }
        return str;
    }

private:
    void transformObjects(const Eigen::Affine3f& cam_pose, sl::Objects& camera_objects);
    void transformBodies(const Eigen::Affine3f& cam_pose, sl::Bodies& camera_bodies);

    int zed_id, zed_sn;
    int topView;
    int cornerView;
    int monitor_topView;
    int monitor_botView;

    sl::Camera zed;
    sl::InitParameters init_parameters;
    sl::BodyTrackingParameters body_tracking_parameters;
    sl::BodyTrackingRuntimeParameters body_runtime_parameters;
    sl::ObjectDetectionParameters object_detection_parameters;
    sl::ObjectDetectionRuntimeParameters object_detection_runtime_parameters;

    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;

    sl::Bodies bodies;
    sl::Objects objects;
    Eigen::Affine3f cam_pose;
};

#endif