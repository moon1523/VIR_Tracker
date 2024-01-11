
#include "MonitorTracker.hh"
#include "FaceDetector.hh"
#include "BodyTracker.hh"
#include "GlassTracker.hh"
#include "utils.hpp"
#include "GLViewer.hpp"

#include <boost/asio.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>

// TODO

void PrintUsage()
{
    cout << endl;
    cout << "Tracking options:" << endl;
    cout << "  --monitor" << endl;
    cout << "  --face" << endl;
    cout << "  --pose [1/2/3/4 - ceiling/monitor_top/monitor_bot/corner]" << endl;
    cout << "  --glass" << endl;
    cout << "  --ocr" << endl;
    cout << endl;
}

int main(int argc, char** argv) 
{
    PrintUsage();
    SetCtrlHandler();
    if (argc < 2) {
        return EXIT_SUCCESS;
    }
    string option(argv[1]);
    cout << option << " is selected." << endl << endl;
    int zed_sn_opt = 1;
    if (option == "--pose" && argc > 2) {
        zed_sn_opt = atoi(argv[2]);
    }

    // settings
    string sn_zed_ceiling     = "24380057";
    string sn_zed_monitor_bot = "20332044";
    string sn_zed_monitor_top = "29134678";
    string sn_zed_corner      = "21929705";
    string sn_kinect_glass         = "000440922112";
    string sn_kinect_monitor       = "000913194512";
    string sn_webcam_face          = "4";
    string VIR_Data_PATH = string(getenv("VIR_Data"));


    // string VIR_cams_json = VIR_Data_PATH + "record/1031/VIR_cams.json";
    // map<string, Eigen::Affine3f> camera_init_pose;
    // VIR_camera_pose(VIR_cams_json, camera_init_pose);
    // string monitor_camParam  = VIR_Data_PATH + "camera/" + sn_kinect_monitor + "_QHD_intrinsic.yml";
    // string monitor_detParam  = VIR_Data_PATH + "camera/detector_params.yml";
    // string monitor_init_pose = VIR_Data_PATH + "record/1031/take18_picc_88F/monitor_a0.txt";
    // string glass_camParam    = VIR_Data_PATH + "camera/000440922112_QHD_intrinsic.yml";
    // string glass_detParam    = VIR_Data_PATH + "camera/detector_params.yml";
    // string record_monitor    = VIR_Data_PATH + "record/1031/take18_picc_88F/CAM_1.avi";
    // string record_face       = VIR_Data_PATH + "record/1031/take18_picc_88F/CAM_0.avi";
    // string record_zeds       = VIR_Data_PATH + "record/1031/take18_picc_88F/ZED_";
    // string record_glass      = VIR_Data_PATH + "record/1031/take18_picc_88F/CAM_2.avi";
    // string train_model       = VIR_Data_PATH + "face_model/1031_Epoch_17.pt";
    // vector<string> member_id = { "0_Doctor", "1_Assist", "2_Radiographer_M0", 
    //                              "3_Nurse_F0", "4_Nurse_F1", "5_Radiographer_M1", "6_Radiographer_M2" };
    // vector<string> member_phantom = { "M_H170W65", "M_H170W65", "M_H170W65", 
    //                                   "F_H160W55", "F_H160W55", "M_H170W65", "M_H170W65" };
    
    string VIR_cams_json = VIR_Data_PATH + "240110/VIR_cams.json";
    map<string, Eigen::Affine3f> camera_init_pose;
    VIR_camera_pose(VIR_cams_json, camera_init_pose);
    string monitor_camParam  = VIR_Data_PATH + "camera/" + sn_kinect_monitor + "_QHD_intrinsic.yml";
    string monitor_detParam  = VIR_Data_PATH + "camera/detector_params.yml";
    string monitor_init_pose = VIR_Data_PATH + "240110/monitor_a0.txt";
    string glass_camParam    = VIR_Data_PATH + "camera/" + sn_kinect_glass + "_QHD_intrinsic.yml";
    string glass_detParam    = VIR_Data_PATH + "camera/detector_params.yml";
    string record_monitor;
    string record_face;
    string record_glass;
    string train_model       = VIR_Data_PATH + "240110/0110.pt";
    vector<string> member_id = { "0_sungho" };
    vector<string> member_phantom = { "M_H170W65" };
    
    boost::asio::io_context io_context;
    boost::asio::ip::udp::socket socket(io_context);
    socket.open(boost::asio::ip::udp::v4());
    boost::asio::ip::udp::resolver resolver(io_context);
    boost::asio::ip::udp::resolver::results_type endpoints = resolver.resolve("127.0.0.1", "12345"); // ip, port

    auto Tracking_Monitor = [&]()->void
    {
        MonitorTracker* MT = new MonitorTracker(record_monitor);
        MT->Set_MonitorPose0(monitor_init_pose);
        MT->Set_CameraPose(camera_init_pose[sn_kinect_monitor]); // translation (meter)
        MT->Set_Parameters(monitor_camParam, monitor_detParam);
        cv::VideoCapture cap = MT->Get_VideoCapture();
        cv::Mat image;
        char key = ' ';
        cout << "[MONITOR] Start communication..." << endl;
        while(!exit_app)
        {
            cap >> image;
            MT->Run(image);
            string tq = MT->Get_Monitor_Pose_as_String(); // tx,ty,tz,qx,qy,qz,qw (t: meter)
            socket.send_to(boost::asio::buffer(tq), *endpoints.begin());
            MT->Render();
            key = (char)cv::waitKey(1);
            if (key == 'q')
                break;
        }
        cout << "[MONITOR] End" << endl;
        cap.release();
        cv::destroyAllWindows();
        delete MT;
    };

    auto Tracking_Face = [&]()->void
    {
        FaceDetector* FD = new FaceDetector(record_face);
        FD->Set_TrainModel(train_model);
        FD->Set_MemberIDs(member_id);
        FD->Initialize();

        cout << "[FACE] Start communication..." << endl;
        while (!exit_app)
        {
            FD->Run(true);
            string faceInfo = FD->Get_FaceInfo_as_String(); // name, similarity, xyz(px)
            socket.send_to(boost::asio::buffer(faceInfo), *endpoints.begin());
        }
        cout << "[FACE] End" << endl;
        delete FD;
    };

    auto Tracking_Pose = [&]()->void
    {
        string sn;
        if      (zed_sn_opt == 1) sn = sn_zed_ceiling;
        else if (zed_sn_opt == 2) sn = sn_zed_monitor_top;
        else if (zed_sn_opt == 3) sn = sn_zed_monitor_bot;
        else if (zed_sn_opt == 4) sn = sn_zed_corner;
        
        BodyTracker* BT = new BodyTracker(sn);
        BT->Set_ceiling(sn_zed_ceiling);
        BT->Set_corner(sn_zed_corner);
        BT->Set_monitor_top(sn_zed_monitor_top);
        BT->Set_monitor_bot(sn_zed_monitor_bot);
        BT->Open();

        GLViewer viewer;
        viewer.init(argc, argv);
        cout << "[POSE] Start communication..." << endl;
        while (!exit_app)
        {
            BT->Run();
            string poses = BT->Get_Pose_As_String(); // head, body data (meter)
            socket.send_to(boost::asio::buffer(poses), *endpoints.begin());
            sl::Objects    cam_objects = BT->Get_Objects();
            sl::Bodies     cam_bodies = BT->Get_Bodies();
            sl::Mat        cam_pointCloud = BT->Get_PointCloud();
            viewer.updateData(cam_objects, cam_bodies, cam_pointCloud);
            viewer.isAvailable();
        }
        viewer.exit();
        cout << "[POSE] End" << endl;
        delete BT;
    };

    auto Tracking_Glass = [&]()->void
    {
        GlassTracker* GT = new GlassTracker();
        GT->Set_CameraPose(camera_init_pose[sn_kinect_glass]); // translation (meter)
        GT->Set_Parameters(glass_camParam, glass_detParam);
        cv::VideoCapture cap = GT->Get_VideoCapture();
        cv::Mat image;
        char key = ' ';
        cout << "[GLASS] Start communication..." << endl;
        while(!exit_app)
        {
            cap >> image;
            GT->Run(image);
            string tq = GT->Get_Glass_Pose_as_String(); // tx,ty,tz,qx,qy,qz,qw (t: meter)
            socket.send_to(boost::asio::buffer(tq), *endpoints.begin());
            GT->Render();
            key = (char)cv::waitKey(1);
            if (key == 'q')
                break;
        }
        cout << "[GLASS] End" << endl;
        cap.release();
        cv::destroyAllWindows();
        delete GT;
    };

    auto Tracking_OCR = [&]()->void
    {


    };

    // execute
    if (option == "--monitor") {
        Tracking_Monitor();
    } else if (option == "--face") {
        Tracking_Face();
    } else if (option == "--pose") {
        Tracking_Pose();
    } else if (option == "--glass") {
        Tracking_Glass();
    } else if (option == "--ocr") {
        Tracking_OCR();
    } else {
        cout << "Invalid option: " << option << endl;
        return EXIT_FAILURE;
    }
    cout << "EXIT_SUCCESS" << endl;
    return EXIT_SUCCESS;


}