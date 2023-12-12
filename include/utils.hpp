#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <signal.h>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>
#include <Eigen/Dense>

using namespace std;

static bool exit_app = false;
static void nix_exit_handler(int s) {
    exit_app = true;
}
// Set the function to handle the CTRL-C
static void SetCtrlHandler() {
    struct sigaction sigIntHandler;
    sigIntHandler.sa_handler = nix_exit_handler;
    sigemptyset(&sigIntHandler.sa_mask);
    sigIntHandler.sa_flags = 0;
    sigaction(SIGINT, &sigIntHandler, NULL);
}

static void VIR_camera_pose(string VIR_cams_json, map<string, Eigen::Affine3f>& camera_init_pose) {
    ifstream ifs(VIR_cams_json);
    if (!ifs.is_open()) {
        cerr << VIR_cams_json << " is not opened" << endl;
        exit(1);
    }
    string dump, dump_prev;
    while(getline(ifs,dump)) {
        stringstream ss(dump);
        ss >> dump;
        if (dump == "\"input\":") {
            string sn = dump_prev.substr(1, dump_prev.size()-3);
            Eigen::Vector3f rvec, tvec;
            while (getline(ifs, dump)) {
                stringstream sa(dump);
                sa >> dump;
                if (dump == "\"rotation\":") {
                    getline(ifs, dump); float rx = stof(dump.substr(0,dump.size()-1));
                    getline(ifs, dump); float ry = stof(dump.substr(0,dump.size()-1));
                    getline(ifs, dump); float rz = stof(dump);
                    rvec = Eigen::Vector3f(rx,ry,rz);
                }
                if (dump == "\"translation\":") {
                    getline(ifs, dump); float tx = stof(dump.substr(0,dump.size()-1));
                    getline(ifs, dump); float ty = stof(dump.substr(0,dump.size()-1));
                    getline(ifs, dump); float tz = stof(dump);
                    tvec = Eigen::Vector3f(tx,ty,tz);
                    break;
                }
            }
            Eigen::Affine3f a = Eigen::Affine3f::Identity();
            a.translate(tvec);
            a.rotate( (Eigen::AngleAxisf(rvec.norm(), rvec.normalized())).toRotationMatrix() );;
            camera_init_pose[sn] = a;
        }
        dump_prev = dump;
    }
    ifs.close();
}

static int findCameraIndex()
{
	string device_list;
  FILE* pipe = popen("v4l2-ctl --list-devices", "r");
  if (pipe == nullptr) {
      cerr << "Error: Could not execute v4l2-ctl command." << endl;
      exit(1);
  }

  char buffer[128];
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      device_list += buffer;
  }
  pclose(pipe);
  stringstream ss(device_list);
  string line, dump;
  int cam_id;
  while (ss >> dump) {      
    if (dump == "Azure") {
        getline(ss, line);
        getline(ss, line);
        stringstream ss2(line);
        ss2 >> dump;
        cam_id = stoi(dump.substr(string("/dev/video").size(), dump.size()));
        break;
    }
  }
  return cam_id;
}

static Eigen::Vector3f averaging_translations(const std::vector<Eigen::Vector3f>& translations)
{
	Eigen::Vector3f t(0,0,0);
	for (auto itr: translations) {
		t += itr;
	}
  t /= (float)translations.size();
	return t;
}

static Eigen::Vector4f averaging_quaternions(const std::vector<Eigen::Vector4f>& quaternions)
{
    if (quaternions.empty())
        return Eigen::Vector4f(0,0,0,1);

    // first build a 4x4 matrix which is the elementwise sum of the product of each quaternion with itself
    Eigen::Matrix4f A = Eigen::Matrix4f::Zero();

    for (int q = 0; q < quaternions.size(); ++q)
        A += quaternions[q] * quaternions[q].transpose();

    // normalise with the number of quaternions
    A /= quaternions.size();

    // Compute the SVD of this 4x4 matrix
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeThinU | Eigen::ComputeThinV);

    Eigen::VectorXf singularValues = svd.singularValues();
    Eigen::MatrixXf U = svd.matrixU();

    // find the eigen vector corresponding to the largest eigen value
    int largestEigenValueIndex;
    float largestEigenValue;
    bool first = true;

    for (int i = 0; i < singularValues.rows(); ++i)
    {
        if (first)
        {
            largestEigenValue = singularValues(i);
            largestEigenValueIndex = i;
            first = false;
        }
        else if (singularValues(i) > largestEigenValue)
        {
            largestEigenValue = singularValues(i);
            largestEigenValueIndex = i;
        }
    }
    Eigen::Vector4f average;
    average(0) = -U(0, largestEigenValueIndex);
    average(1) = -U(1, largestEigenValueIndex);
    average(2) = -U(2, largestEigenValueIndex);
    average(3) = -U(3, largestEigenValueIndex);

    return average;
}

static Eigen::Affine3f averaging_Affine3f(const vector<Eigen::Affine3f>& affs)
{
	vector<Eigen::Vector4f> Qs;
	vector<Eigen::Vector3f> Ts;
	for (auto itr: affs) {
		Qs.push_back(Eigen::Vector4f( Eigen::Quaternionf(itr.linear()).coeffs() ));
		Ts.push_back(Eigen::Vector3f( itr.translation() ));
	}
	Eigen::Vector4f q = averaging_quaternions(Qs);
	Eigen::Vector3f t = averaging_translations(Ts);
	Eigen::Affine3f a;
	a.linear() = Eigen::Quaternionf(q).normalized().toRotationMatrix();
	a.translation() = t;
	return a;
}

#endif