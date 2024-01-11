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

    void Set_ceiling(string sn) { ceiling = stoi(sn); }
    void Set_corner(string sn) { corner = stoi(sn); }
    void Set_monitor_top(string sn) { monitor_top = stoi(sn); }
    void Set_monitor_bot(string sn) { monitor_bot = stoi(sn); }

    sl::Transform Get_CameraPose_SL() { return cam_pose_sl; }
    sl::Objects   Get_Objects()       { return objects; }
    sl::Bodies    Get_Bodies()        { return bodies; }
    sl::Mat       Get_PointCloud()    { return point_cloud; }

    string Get_Pose_As_String() {
        string str;
        str += zed_sn2name[zed_sn] + "\n";
        if (zed_sn == ceiling || zed_sn == corner) {
            for (auto &head : objects.object_list) {
                str += "head_id " + to_string(head.id) + "\n";
                str += to_string(head.position.x) + " " + to_string(head.position.y) + " " + to_string(head.position.z) + " " + to_string(head.confidence*0.01) + "\n";
            }
        }
        for (auto &body : bodies.body_list) {
            str += "body_id " + to_string(body.id) + "\n";
            int count = 0;
            for (auto &kpt : body.keypoint) {
                str += to_string(kpt.x) + " " + to_string(kpt.y) + " " + to_string(kpt.z) + " " + to_string(body.keypoint_confidence[count++]*0.01) + "\n";
            }
        }
        return str;
    }

private:
    void filteringObjects(sl::Objects& camera_objects);
    void filteringBodies(sl::Bodies& camera_bodies);
    void bboxInfo(const std::vector<sl::float3>& bbox, sl::float3& center, float& edge1, float& edge2, float& edge3);

    int zed_id, zed_sn;
    map<int, string> zed_sn2name;
    int ceiling;
    int corner;
    int monitor_top;
    int monitor_bot;

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