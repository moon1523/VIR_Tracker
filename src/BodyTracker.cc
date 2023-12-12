#include "BodyTracker.hh"

static bool checkZEDState(const sl::ERROR_CODE& state) {
    if (state != sl::ERROR_CODE::SUCCESS) {
        std::cout << "Error: " << state << std::endl;
        return false;
    }
    return true;
}

BodyTracker::BodyTracker(string sn)
{
    // check serial number
    vector<sl::DeviceProperties> devices = sl::Camera::getDeviceList();
    for (auto z: devices) {
        if (stoi(sn) == z.serial_number) {
            zed_sn = z.serial_number;
            zed_id = z.id;
            break;
        }
    }
    cout << "Seleted zed sn (id): " << zed_sn << " (" << zed_id << ")" << endl;
}

BodyTracker::~BodyTracker()
{
    zed.close();


}

bool BodyTracker::Open()
{
    // open the camera
    init_parameters.input.setFromCameraID(zed_id);
    init_parameters.camera_resolution = sl::RESOLUTION::HD720;
    init_parameters.depth_mode = sl::DEPTH_MODE::ULTRA;
    init_parameters.coordinate_system = sl::COORDINATE_SYSTEM::IMAGE;
    init_parameters.coordinate_units = sl::UNIT::METER;
    init_parameters.depth_maximum_distance = 10.0f;
    init_parameters.camera_fps = 15;

    auto state = zed.open(init_parameters);
    if(!checkZEDState(state)) return false;

    // in most cases in body tracking setup, the cameras are static
    sl::PositionalTrackingParameters positional_tracking_parameters;
    positional_tracking_parameters.set_as_static = false;
    state = zed.enablePositionalTracking(positional_tracking_parameters);
    if(!checkZEDState(state)) return false;

    // define the body tracking parameters, as the fusion can does the tracking and fitting you don't need to enable them here, unless you need it for your app
    body_tracking_parameters.body_format = sl::BODY_FORMAT::BODY_34;
    body_tracking_parameters.enable_tracking = true;
    body_tracking_parameters.enable_body_fitting = true;
    body_tracking_parameters.enable_segmentation = false; // designed to give person pixel mask
    body_tracking_parameters.detection_model = sl::BODY_TRACKING_MODEL::HUMAN_BODY_ACCURATE;
    body_tracking_parameters.instance_module_id = 0; // select instance ID
    body_runtime_parameters.detection_confidence_threshold = 60;
    state = zed.enableBodyTracking(body_tracking_parameters);
    if(!checkZEDState(state)) return false;

    sl::CalibrationParameters calibration_parameters 
        = zed.getCameraInformation().camera_configuration.calibration_parameters_raw;
    double fx = calibration_parameters.left_cam.fx;
    double fy = calibration_parameters.left_cam.fy;
    double cx = calibration_parameters.left_cam.cx;
    double cy = calibration_parameters.left_cam.cy;
    double k1 = calibration_parameters.left_cam.disto[0];
    double k2 = calibration_parameters.left_cam.disto[1];
    double p1 = calibration_parameters.left_cam.disto[2];
    double p2 = calibration_parameters.left_cam.disto[3];
    double k3 = calibration_parameters.left_cam.disto[4];
    double k4 = calibration_parameters.left_cam.disto[5];
    double k5 = calibration_parameters.left_cam.disto[6];
    double k6 = calibration_parameters.left_cam.disto[7];
    double s1 = calibration_parameters.left_cam.disto[8];
    double s2 = calibration_parameters.left_cam.disto[9];
    double s3 = calibration_parameters.left_cam.disto[10];
    double s4 = calibration_parameters.left_cam.disto[11];
    cameraMatrix = (cv::Mat_<double>(3,3)  << fx,0,cx, 0,fy,cy, 0,0,1);
    distCoeffs   = (cv::Mat_<double>(1,12) << k1,k2,p1,p2,k3,k4,k5,k6,s1,s2,s3,s4);

    object_detection_parameters.enable_tracking = true;
    object_detection_parameters.enable_segmentation = false; // designed to give person pixel mask
    object_detection_parameters.detection_model = sl::OBJECT_DETECTION_MODEL::PERSON_HEAD_BOX_ACCURATE;
    object_detection_parameters.instance_module_id = 1; // select instance ID
    object_detection_runtime_parameters.detection_confidence_threshold = 40;
    object_detection_runtime_parameters.object_class_filter = { sl::OBJECT_CLASS::PERSON };
    state = zed.enableObjectDetection(object_detection_parameters);
    if(!checkZEDState(state)) return false;
    return true;
}

void BodyTracker::Run()
{
    sl::Pose pose;
    if (zed.grab() != sl::ERROR_CODE::SUCCESS)
        return;
    if (zed_sn == topView) {
        zed.retrieveObjects(objects, object_detection_runtime_parameters, object_detection_parameters.instance_module_id);
        transformObjects(cam_pose, objects);
    }
    zed.retrieveBodies(bodies, body_runtime_parameters, body_tracking_parameters.instance_module_id);
    transformBodies(cam_pose, bodies);
    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, sl::Resolution(1280, 720));
}

void BodyTracker::transformObjects(const Eigen::Affine3f& cam_pose, sl::Objects& camera_objects)
{
    for (auto &head : camera_objects.object_list)
    {
        Eigen::Vector3f head_tf = cam_pose * Eigen::Vector3f(head.head_position.x, head.head_position.y, head.head_position.z);
        head.head_position.x = head_tf.x();
        head.head_position.y = head_tf.y();
        head.head_position.z = head_tf.z();

        for (auto &bb : head.head_bounding_box)
        {
            Eigen::Vector3f bb_tf = cam_pose * Eigen::Vector3f(bb.x, bb.y, bb.z);
            bb.x = bb_tf.x();
            bb.y = bb_tf.y();
            bb.z = bb_tf.z();
        }
    }
}
void BodyTracker::transformBodies(const Eigen::Affine3f& cam_pose, sl::Bodies& camera_bodies)
{
    for (auto &body : camera_bodies.body_list)
    {
        for (auto &kpt : body.keypoint)
        {
            Eigen::Vector3f kpt_tf = cam_pose * Eigen::Vector3f(kpt.x, kpt.y, kpt.z);
            kpt.x = kpt_tf.x();
            kpt.y = kpt_tf.y();
            kpt.z = kpt_tf.z();
        }
        for (auto &pt : body.bounding_box)
        {
            Eigen::Vector3f pt_tf = cam_pose * Eigen::Vector3f(pt.x, pt.y, pt.z);
            pt.x = pt_tf.x();
            pt.y = pt_tf.y();
            pt.z = pt_tf.z();
        }
    }
}