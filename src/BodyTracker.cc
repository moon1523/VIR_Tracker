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

    bool isFind(false);
    for (auto z: devices) {
        if (stoi(sn) == z.serial_number) {
            zed_sn = z.serial_number;
            zed_id = z.id;
            isFind = true;
            break;
        }
    }
    if (!isFind) {
        cout << "sn you need: " << sn << endl;
        cout << "connected sn list" << endl;
        for (auto z: devices) cout << z.serial_number << endl;
        exit(1);
    }
}

BodyTracker::~BodyTracker()
{
    zed.close();
}

bool BodyTracker::Open()
{
    cout << "Seleted zed sn (id): " << zed_sn << " (" << zed_id << ")" << endl;
    zed_sn2name[ceiling] = "ceiling";
    zed_sn2name[corner] = "corner";
    zed_sn2name[monitor_top] = "monitor_top";
    zed_sn2name[monitor_bot] = "monitor_bot";

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
    if (zed_sn == ceiling || zed_sn == corner) {
        zed.retrieveObjects(objects, object_detection_runtime_parameters, object_detection_parameters.instance_module_id);
        filteringObjects(objects);
    }
    zed.retrieveBodies(bodies, body_runtime_parameters, body_tracking_parameters.instance_module_id);
    filteringBodies(bodies);
    zed.retrieveMeasure(point_cloud, sl::MEASURE::XYZRGBA, sl::MEM::GPU, sl::Resolution(1280, 720));
}

void BodyTracker::filteringObjects(sl::Objects& camera_objects)
{    
    for (auto it = camera_objects.object_list.begin(); it != camera_objects.object_list.end(); ) {
        sl::float3 center; float edge1, edge2, edge3;
        bboxInfo(it->head_bounding_box, center, edge1, edge2, edge3);
        it->position = center;
        // 1. check tracking state
        if (it->tracking_state != sl::OBJECT_TRACKING_STATE::OK) {
            it = camera_objects.object_list.erase(it);
        }
        // 2. check speed
        else if (it->velocity.norm() > 1.5) {
            it = camera_objects.object_list.erase(it);
        }
        // 3. check bounding box size
        else if (it->head_bounding_box.size() != 8) {
            it = camera_objects.object_list.erase(it);
        }
        // 4. check bounding box
        else if (edge1 < 0.1 || edge1 > 0.3 ||
                 edge2 < 0.1 || edge2 > 0.3 ||
                 edge3 < 0.1 || edge3 > 0.3) {
            it = camera_objects.object_list.erase(it);
        }
        else {
            it++;
        }
    }
}

void BodyTracker::filteringBodies(sl::Bodies& camera_bodies)
{
    for (auto it = camera_bodies.body_list.begin(); it != camera_bodies.body_list.end(); ) {
        sl::float3 center; float edge1, edge2, edge3;
        bboxInfo(it->bounding_box, center, edge1, edge2, edge3);
        it->position = center;
        // 1. check tracking state
        if (it->tracking_state != sl::OBJECT_TRACKING_STATE::OK) {
            it = camera_bodies.body_list.erase(it);
        }
        // 2. check speed
        else if (it->velocity.norm() > 1.5) {
            it = camera_bodies.body_list.erase(it);
        }
        // 3. check neck-head length
        else if ( (it->keypoint[26] - it->keypoint[3]).norm() > 0.3f ) {
            it = camera_bodies.body_list.erase(it);
        }
        else {
            it++;
        }
    }
}

void BodyTracker::bboxInfo(const std::vector<sl::float3>& bbox, sl::float3& center, float& edge1, float& edge2, float& edge3)
{
    if (bbox.size() != 8) return;
    //    1 ------ 2
    //   /        /|
    //  0 ------ 3 |
    //  | Object | 6
    //  |        |/
    //  4 ------ 7
    center = (bbox[0] + bbox[1] + bbox[2] + bbox[3] + bbox[4] + bbox[5] + bbox[6] + bbox[7]) * 0.125f;
    edge1 = (bbox[0] - bbox[1]).norm();
    edge2 = (bbox[0] - bbox[3]).norm();
    edge3 = (bbox[0] - bbox[4]).norm();
}
