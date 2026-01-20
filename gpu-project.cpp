#include <iostream>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include "cudacut.cuh"

using namespace cv;
using namespace std;

static double g_gpu_peak_used_mb = 0.0;
static double g_gpu_total_mb = 0.0;

// Cập nhật đỉnh sử dụng bộ nhớ GPU (silent)
static void printGpuMemUsage(const string &tag) {
	size_t free_bytes = 0, total_bytes = 0;
	cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
	if (err == cudaSuccess) {
		double free_mb = free_bytes / 1024.0 / 1024.0;
		double total_mb = total_bytes / 1024.0 / 1024.0;
		double used_mb = total_mb - free_mb;
		if (g_gpu_total_mb <= 0.0) g_gpu_total_mb = total_mb;
		if (used_mb > g_gpu_peak_used_mb) g_gpu_peak_used_mb = used_mb;
		(void)tag;
	} else {
		(void)tag; (void)err;
	}
}

static void showGpuMemPeak() {
	if (g_gpu_total_mb <= 0.0) {
		printGpuMemUsage("peak-sample");
	}
	double used_mb = g_gpu_peak_used_mb;
	double total_mb = g_gpu_total_mb > 0.0 ? g_gpu_total_mb : 0.0;
	double used_pct = (total_mb > 0.0) ? (used_mb * 100.0 / total_mb) : 0.0;
	std::ios::fmtflags old_flags = cout.flags();
	std::streamsize old_prec = cout.precision();
	cout.setf(ios::fixed);
	cout.precision(1);
	cout << "GPU peak used (overall): " << used_mb << " MB (" << used_pct << "% of "
		 << total_mb << " MB)" << endl;
	cout.flags(old_flags);
	cout.precision(old_prec);
}

// GPU helpers
extern int autoDetectOverlapGPU(const Mat& A, const Mat& B);
extern Mat multiBandBlendGPU(const Mat& imgA, const Mat& imgB, const Mat& mask, int levels);


struct SeamInfo {
	int seam_x;
	int overlap_width;
	Mat seam_region;
};


vector<SeamInfo> g_seams;
Mat g_result_raw;      
Mat g_result_stitched; 

Mat stitchTwoImages(Mat A, Mat B, Mat& raw_result, int pair_index, bool verbose = true) {
	if (A.empty()) return B.clone();
	if (B.empty()) return A.clone();
	
	if (A.rows != B.rows) {
		cout << "  ERROR: Height mismatch! A=" << A.rows << ", B=" << B.rows << endl;
		return A.clone();
	}
	
	if (verbose && A.cols != B.cols) {
		cout << "  Width: A=" << A.cols << ", B=" << B.cols << " (different OK)" << endl;
	}
	
	// Chuyển sang grayscale cho phần graph-cut
	bool is_color = (A.channels() == 3);
	int img_type = is_color ? CV_8UC3 : CV_8UC1;
	Mat A_gray, B_gray;
	if (is_color) {
		cvtColor(A, A_gray, COLOR_BGR2GRAY);
		cvtColor(B, B_gray, COLOR_BGR2GRAY);
	} else {
		A_gray = A;
		B_gray = B;
	}
	
	printGpuMemUsage("before autoDetectOverlapGPU");
	int overlap_width = autoDetectOverlapGPU(A_gray, B_gray);
	printGpuMemUsage("after autoDetectOverlapGPU");
	if (verbose) {
		cout << "  Overlap: " << overlap_width << " px" << endl;
	}
	
	
	int result_width = A.cols + B.cols - overlap_width;
	Mat result(A.rows, result_width, img_type, Scalar::all(0));
	Mat result_stitched(A.rows, result_width, img_type, Scalar::all(0));
	
	
	A.copyTo(result(cv::Rect(0, 0, A.cols, A.rows)));
	B.copyTo(result(cv::Rect(A.cols - overlap_width, 0, B.cols, B.rows)));
	
	
	raw_result = result.clone();
	
	
	printGpuMemUsage("before graphcut ctor");
	CudaCut graphcut(A.cols, A.rows, overlap_width);
	//Global_relabel
	graphcut.use_gpu_global_relabel = false; 
	graphcut.cudaCutsInit();
	printGpuMemUsage("after cudaCutsInit");
	graphcut.cudaCutsSetupGraph(A_gray, B_gray);
	graphcut.push_relabel();
	
	
	if (is_color) {
		Mat gray_result(A.rows, result_width, CV_8UC1, Scalar(0));
		Mat gray_stitched(A.rows, result_width, CV_8UC1, Scalar(0));
		A_gray.copyTo(gray_result(cv::Rect(0, 0, A.cols, A.rows)));
		B_gray.copyTo(gray_result(cv::Rect(A.cols - overlap_width, 0, B.cols, B.rows)));
		
		
		printGpuMemUsage("before selectPix (color path)");
		graphcut.selectPix(gray_result, gray_stitched);
		printGpuMemUsage("after selectPix (color path)");
		
		
		A(cv::Rect(0, 0, A.cols - overlap_width, A.rows))
			.copyTo(result_stitched(cv::Rect(0, 0, A.cols - overlap_width, A.rows)));
		
		B(cv::Rect(overlap_width, 0, B.cols - overlap_width, B.rows))
			.copyTo(result_stitched(cv::Rect(A.cols, 0, B.cols - overlap_width, B.rows)));
		
		
		int xoffset = A.cols - overlap_width;
		for (int y = 0; y < A.rows; y++) {
			for (int x = 0; x < overlap_width; x++) {
				int idx = y * overlap_width + x;
				if (graphcut.scanned[idx]) {
					result_stitched.at<Vec3b>(y, xoffset + x) = B.at<Vec3b>(y, x);
				} else {
					result_stitched.at<Vec3b>(y, xoffset + x) = A.at<Vec3b>(y, xoffset + x);
				}
			}
		}
		
		// Đánh dấu đường seam để hiển thị
		for (int y = 0; y < A.rows; y++) {
			for (int x = 1; x < overlap_width - 1; x++) {
				int idx = y * overlap_width + x;
				if (graphcut.scanned[idx] != graphcut.scanned[idx + 1]) {
					result_stitched.at<Vec3b>(y, xoffset + x) = Vec3b(255, 255, 255);
					result_stitched.at<Vec3b>(y, xoffset + x + 1) = Vec3b(255, 255, 255);
					break;
				}
			}
		}
	} else {
		
		printGpuMemUsage("before selectPix (grayscale path)");
		graphcut.selectPix(result, result_stitched);
		printGpuMemUsage("after selectPix (grayscale path)");
	}
	
	
	SeamInfo seam;
	seam.seam_x = A.cols - overlap_width;
	seam.overlap_width = overlap_width;
	seam.seam_region = result_stitched(cv::Rect(seam.seam_x, 0, overlap_width, result_stitched.rows)).clone();
	

	for (int y = 0; y < A.rows; y++) {
		for (int x = 1; x < overlap_width - 1; x++) {
			int idx = y * overlap_width + x;
			if (graphcut.scanned[idx] != graphcut.scanned[idx + 1]) {
				if (is_color) {
					seam.seam_region.at<Vec3b>(y, x) = Vec3b(0, 0, 255);
					seam.seam_region.at<Vec3b>(y, x + 1) = Vec3b(0, 0, 255);
				} else {
					seam.seam_region.at<uchar>(y, x) = 255;
					seam.seam_region.at<uchar>(y, x + 1) = 255;
				}
				break;
			}
		}
	}
	
	g_seams.push_back(seam);
	
	graphcut.cudaCutsFreeMem();
	printGpuMemUsage("after cudaCutsFreeMem");
	
	return result_stitched;
}

Mat stitchMultipleImages(const vector<Mat>& images) {
	if (images.empty()) return Mat();
	if (images.size() == 1) return images[0].clone();
	
	bool is_color = (images[0].channels() == 3);
	cout << "\n=== Stitching " << images.size() << " images ===" << endl;
	cout << "Mode: " << (is_color ? "COLOR (BGR)" : "GRAYSCALE") << endl;
	g_seams.clear();
	
	Mat result = images[0].clone();
	Mat raw_result;
	vector<int> overlaps;
	
	for (size_t i = 1; i < images.size(); i++) {
		cout << "\n[" << i << "/" << (images.size()-1) << "] Merging image " << (i+1) << endl;
		cout << "  Panorama: " << result.cols << "x" << result.rows << endl;
		cout << "  + Image:  " << images[i].cols << "x" << images[i].rows << endl;
		
		Mat result_gray, img_gray;
		if (is_color) {
			cvtColor(result, result_gray, COLOR_BGR2GRAY);
			cvtColor(images[i], img_gray, COLOR_BGR2GRAY);
		} else {
			result_gray = result;
			img_gray = images[i];
		}
		
		auto detect_start = getMoment;
		int overlap_width = autoDetectOverlapGPU(result_gray, img_gray);
		auto detect_end = getMoment;
		overlaps.push_back(overlap_width);
		cout << "  Overlap detected: " << overlap_width << " px (" 
			 << TimeCpu(detect_end, detect_start) / 1000.0 << " ms)" << endl;
		
		auto start = getMoment;
		
		Mat temp_raw;
		result = stitchTwoImages(result, images[i], temp_raw, (int)i, false);
		auto end = getMoment;
		
		int graph_nodes = overlap_width * result.rows;
		cout << "  Graph size: " << overlap_width << "x" << result.rows 
			 << " = " << graph_nodes << " nodes" << endl;
		cout << "  = Result: " << result.cols << "x" << result.rows << endl;
		cout << "  Graph cut time: " << TimeCpu(end, start) / 1000.0 << " ms" << endl;
	}
	
	// Tạo kết quả multi-band blending trên GPU
	cout << "\n  Creating multi-band blended result (GPU)..." << endl;
	printGpuMemUsage("before multiBandBlend loop");
	auto blend_start_time = getMoment;
	raw_result = images[0].clone();
	int img_type = is_color ? CV_8UC3 : CV_8UC1;
	
	for (size_t i = 1; i < images.size(); i++) {
		int overlap = overlaps[i-1];
		int new_width = raw_result.cols + images[i].cols - overlap;
		
		Mat imgA = Mat::zeros(raw_result.rows, new_width, img_type);
		Mat imgB = Mat::zeros(raw_result.rows, new_width, img_type);
		Mat mask = Mat::zeros(raw_result.rows, new_width, CV_8UC1);
		
		raw_result.copyTo(imgA(cv::Rect(0, 0, raw_result.cols, raw_result.rows)));
		images[i].copyTo(imgB(cv::Rect(raw_result.cols - overlap, 0, images[i].cols, images[i].rows)));
		int blend_start = raw_result.cols - overlap;
		for (int y = 0; y < mask.rows; y++) {
			for (int x = 0; x < blend_start; x++) {
				mask.at<uchar>(y, x) = 0;
			}
			for (int x = 0; x < overlap; x++) {
				mask.at<uchar>(y, blend_start + x) = (uchar)(255.0 * x / overlap);
			}
			for (int x = raw_result.cols; x < new_width; x++) {
				mask.at<uchar>(y, x) = 255;
			}
		}
		
		// Tính số mức pyramid dựa trên kích thước ảnh
		int min_dim = min(imgA.rows, imgA.cols);
		int levels = min(6, (int)log2(min_dim / 16));
		levels = max(2, levels);
		
		printGpuMemUsage("before multiBandBlendGPU call");
		raw_result = multiBandBlendGPU(imgA, imgB, mask, levels);
		printGpuMemUsage("after multiBandBlendGPU call");
	}
	auto blend_end_time = getMoment;
	cout << "  Multi-band blend time: " << TimeCpu(blend_end_time, blend_start_time) / 1000.0 << " ms" << endl;
	
	g_result_raw = raw_result;
	
	return result;
}

int main(int argc, char** argv)
{

	vector<string> paths;
	if (argc <= 1) {
		// default small test set (fast)
		paths = {
			"D:\\pic\\image_1_1008x755.png",
			"D:\\pic\\image_2_1008x755.png"
		};
	} else {
		string arg1 = argv[1];
		if (arg1 == "-h" || arg1 == "--help" || arg1 == "help") {
			cout << "Usage:\n";
			cout << "  " << argv[0] << " [preset|--files file1 file2 ...|file1 file2 ...]\n";
			cout << "Presets: default, 2k, 4k, 8k\n";
			return 0;
		}

		if (arg1 == "--files") {
			for (int i = 2; i < argc; ++i) paths.emplace_back(argv[i]);
		} else if (arg1 == "2k") {
			paths = {
				"D:\\pic\\2k_1.jpg",
				"D:\\pic\\2k_2.jpg",
				"D:\\pic\\2k_3.jpg",
				"D:\\pic\\2k_4.jpg",
				"D:\\pic\\2k_5.jpg",
				"D:\\pic\\2k_6.jpg"
			};
		} else if (arg1 == "4k") {
			paths = {
				"D:\\pic\\4k_1.jpg",
				"D:\\pic\\4k_2.jpg"
			};
		} else if (arg1 == "8k") {
			paths = {
				"D:\\pic\\8k_1.jpg",
				"D:\\pic\\8k_2.jpg",
				"D:\\pic\\8k_3.jpg",
				"D:\\pic\\8k_4.jpg",
				"D:\\pic\\8k_5.jpg"
			};
		} else {
			// treat all args as file paths
			for (int i = 1; i < argc; ++i) paths.emplace_back(argv[i]);
		}
	}

	cout << "Using " << paths.size() << " input images:\n";
	for (auto &p : paths) cout << "  " << p << "\n";
	
	// Color loading mode: true = BGR (3 channels), false = Grayscale
	bool load_color = true;
	
	// Load images
	vector<Mat> images;
	cout << "=== Loading " << paths.size() << " images ===" << endl;
	cout << "Color mode: " << (load_color ? "BGR (3 channels)" : "Grayscale (1 channel)") << endl;
	
	for (size_t i = 0; i < paths.size(); i++) {
		Mat img = imread(paths[i], load_color ? IMREAD_COLOR : IMREAD_GRAYSCALE);
		if (img.empty()) {
			cout << "[SKIP] Cannot load: " << paths[i] << endl;
			continue;
		}
		cout << "  [" << (i+1) << "] " << img.cols << "x" << img.rows 
			 << " (" << img.channels() << " channels) - " << paths[i] << endl;
		images.push_back(img);
	}
	
	if (images.size() < 2) {
		cout << "Error: Need at least 2 images!" << endl;
		return -1;
	}
	
	// Detect resolution
	string resolution = "SD";
	if (images[0].cols >= 3840 || images[0].rows >= 2160) resolution = "4K";
	else if (images[0].cols >= 2560 || images[0].rows >= 1440) resolution = "2K";
	else if (images[0].cols >= 1920 || images[0].rows >= 1080) resolution = "Full HD";
	else if (images[0].cols >= 1280 || images[0].rows >= 720) resolution = "HD";
	cout << "Detected resolution: " << resolution << endl;
	
	auto total_start = getMoment;
	Mat panorama = stitchMultipleImages(images);
	auto total_end = getMoment;
	
	cout << "\n=== DONE ===" << endl;
	cout << "========================================" << endl;
	cout << "           SUMMARY STATISTICS          " << endl;
	cout << "========================================" << endl;
	cout << "Color mode: " << (images[0].channels() == 3 ? "BGR (3 channels)" : "Grayscale (1 channel)") << endl;
	cout << "Input images: " << images.size() << endl;
	for (size_t i = 0; i < images.size(); i++) {
		cout << "  Image " << (i+1) << ": " << images[i].cols << "x" << images[i].rows 
			 << " (" << (images[i].cols * images[i].rows / 1000000.0) << " MP)" << endl;
	}
	cout << "----------------------------------------" << endl;
	cout << "Final panorama: " << panorama.cols << "x" << panorama.rows 
		 << " (" << (panorama.cols * panorama.rows / 1000000.0) << " MP)" << endl;
	cout << "Resolution: " << resolution << endl;
	cout << "----------------------------------------" << endl;
	cout << "Number of seams: " << g_seams.size() << endl;
	for (size_t i = 0; i < g_seams.size(); i++) {
		cout << "  Seam " << (i+1) << " (pair " << (i+1) << "-" << (i+2) << "): "
			 << "pos=" << g_seams[i].seam_x << "px, "
			 << "overlap=" << g_seams[i].overlap_width << "px" << endl;
	}
	cout << "----------------------------------------" << endl;
	cout << "Total time: " << TimeCpu(total_end, total_start) / 1000.0 << " ms" << endl;
	cout << "Throughput: " << (panorama.cols * panorama.rows) / (TimeCpu(total_end, total_start) / 1000.0) << " pixels/ms" << endl;
	cout << "========================================" << endl;
	showGpuMemPeak();
	
	// Display results
	int max_display_width = 1920;
	int max_display_height = 600;
	int display_width = panorama.cols;
	int display_height = panorama.rows;
	
	if (display_width > max_display_width || display_height > max_display_height) {
		float scale = min((float)max_display_width / display_width, 
		                  (float)max_display_height / display_height);
		display_width = (int)(display_width * scale);
		display_height = (int)(display_height * scale);
	}

	// Multi-band blended result
	cv::namedWindow("Multi-band Blend (No Graph Cut)", cv::WINDOW_NORMAL);
	cv::resizeWindow("Multi-band Blend (No Graph Cut)", display_width, display_height);
	cv::imshow("Multi-band Blend (No Graph Cut)", g_result_raw);
	
	// Graph cut stitching result
	cv::namedWindow("Graph Cut Stitching", cv::WINDOW_NORMAL);
	cv::resizeWindow("Graph Cut Stitching", display_width, display_height);
	cv::imshow("Graph Cut Stitching", panorama);
	
	// Zoom into each seam
	int seam_window_height = 600;
	for (size_t i = 0; i < g_seams.size(); i++) {
		string win_name = "Seam " + to_string(i+1) + " (pair " + to_string(i+1) + "-" + to_string(i+2) + ")";
		
		int seam_width = g_seams[i].seam_region.cols;
		int seam_height = g_seams[i].seam_region.rows;
		int win_width = (int)(seam_width * seam_window_height / (float)seam_height);
		win_width = max(win_width, 100);
		
		cv::namedWindow(win_name, cv::WINDOW_NORMAL);
		cv::resizeWindow(win_name, win_width, seam_window_height);
		cv::imshow(win_name, g_seams[i].seam_region);
		
		cout << "Seam " << (i+1) << ": position=" << g_seams[i].seam_x 
			 << ", width=" << g_seams[i].overlap_width << "px" << endl;
	}
	
	// Save full resolution image
	if (!g_result_raw.empty()) {
		bool ok = cv::imwrite("D:\\pic\\panorama_blend.png", g_result_raw);
		cout << (ok ? "\nSaved: panorama_blend.png" : "\nFailed to save") << endl;
	}
	
	cv::waitKey(0);
	return 0;
}