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
    void Set_CameraPose(const Eigen::Affine3f& pose) { 
        cam_pose = pose;
        Eigen::Vector3f T = pose.translation();
        Eigen::Quaternionf Q = Eigen::Quaternionf( pose.rotation() ); Q.normalize();
        sl::float3 T_sl(T.x(), T.y(), T.z());
        sl::float4 Q_sl(Q.x(), Q.y(), Q.z(), Q.w());
        cam_pose_sl.setTranslation(T_sl);
        cam_pose_sl.setOrientation(Q_sl);
    }
    sl::CameraParameters Get_CameraParameters() { 
        return zed.getCameraInformation().camera_configuration.calibration_parameters.left_cam; 
    }

    sl::Transform Get_CameraPose_SL() { return cam_pose_sl; }
    sl::Objects   Get_Objects()       { return objects; }
    sl::Bodies    Get_Bodies()        { return bodies; }
    sl::Mat       Get_PointCloud()    { return point_cloud; }

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

    Eigen::Affine3f cam_pose;
    sl::Transform cam_pose_sl;
    sl::Bodies bodies;
    sl::Objects objects;
    sl::Mat point_cloud;
    
};

#endif