// gpu-project.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <iomanip>
#include <string>
#include <cuda_runtime.h>
#include "cudacut.cuh"

using namespace cv;
using namespace std;

void CudaCut::selectPix(cv::Mat& result, cv::Mat& result1) {
	cv::Mat tmp(result, cv::Rect(image_width - width, 0, width, height));
	cv::imshow("tmp", tmp);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width - 1; j++) {
			if (scanned[i * width + j] == false) {
				tmp.at<uchar>(i, j) = area1.at<uchar>(i, j);
			}
			else {
				tmp.at<uchar>(i, j) = area2.at<uchar>(i, j);
			}

			/*if (j == 0) {
				tmp.at<uchar>(i, j) = 255;
			}
			if (j == width - 2) {
				tmp.at<uchar>(i, j + 1) = 255;
			}*/

		}
	}
	result.copyTo(result1);
	cv::Mat tmp1(result1, cv::Rect(image_width - width, 0, width, height));
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (j > 0 && j < width - 1 && scanned[i * width + j] != scanned[i * width + j + 1]) {
				tmp1.at<uchar>(i, j) = 255;
				tmp1.at<uchar>(i, j + 1) = 255;
				//tmp1.at<uchar>(i, j - 1) = 255;

			}
			if (i > 0 && i < height - 1 && scanned[i * width + j] != scanned[(i + 1) * width + j]) {
				tmp1.at<uchar>(i, j) = 255;
				tmp1.at<uchar>(i + 1, j) = 255;
				//tmp1.at<uchar>(i - 1, j) = 255;
			}

		}
	}
}

// GPU memory peak sampler (silent) -------------------------------------------------
static double g_gpu_peak_used_mb = 0.0;
static double g_gpu_total_mb = 0.0;

void sampleGpuMemUsage()
{
	size_t free_bytes = 0, total_bytes = 0;
	cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
	if (err == cudaSuccess && total_bytes > 0) {
		double used_mb = double(total_bytes - free_bytes) / (1024.0 * 1024.0);
		double total_mb = double(total_bytes) / (1024.0 * 1024.0);
		if (total_mb > g_gpu_total_mb) g_gpu_total_mb = total_mb;
		if (used_mb > g_gpu_peak_used_mb) g_gpu_peak_used_mb = used_mb;
	}
}

void showGpuMemPeak()
{
	if (g_gpu_total_mb <= 0.0) {
		size_t free_bytes = 0, total_bytes = 0;
		if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess && total_bytes > 0) {
			g_gpu_total_mb = double(total_bytes) / (1024.0 * 1024.0);
		}
	}

	if (g_gpu_peak_used_mb > 0.0) {
		ios::fmtflags f = cout.flags();
		streamsize p = cout.precision();
		cout.setf(std::ios::fixed);
		cout << setprecision(1);
		double percent = (g_gpu_total_mb > 0.0) ? (g_gpu_peak_used_mb / g_gpu_total_mb) * 100.0 : 0.0;
		cout << "GPU peak used (overall): " << g_gpu_peak_used_mb << " MB (" << percent << "% of " << g_gpu_total_mb << " MB)" << "\n";
		cout.flags(f);
		cout.precision(p);
	}
}



int main()
{
	string path = "images/APAP dataset/image_1_1008x755.png";
	string path1 = "images/APAP dataset/image_2_1008x755.png";
	Mat A = imread(path, IMREAD_GRAYSCALE); // 1008 x 755
	Mat B = imread(path1, IMREAD_GRAYSCALE); // 1008 x 755
	
	Mat result(A.rows, A.cols * 2 - OVERLAP_WIDTH, CV_8UC1);
	Mat result1(A.rows, A.cols * 2 - OVERLAP_WIDTH, CV_8UC1);

	A.copyTo(result(cv::Rect(0, 0, A.cols, A.rows)));
	B.copyTo(result(cv::Rect(A.cols - OVERLAP_WIDTH, 0, A.cols, A.rows)));

	CudaCut graphcut(A.cols, A.rows, OVERLAP_WIDTH);
	graphcut.cudaCutsInit();
		// sample GPU memory after initialization
		// (silent sampler updates peak in global)
		// forward declaration defined below
		extern void sampleGpuMemUsage();
		sampleGpuMemUsage();
	auto start = getMoment;
	auto start1 = getMoment;
	graphcut.cudaCutsSetupGraph(A, B);
		sampleGpuMemUsage();
	auto end1 = getMoment;
	cout << "Construct Graph Time = " << TimeCpu(end1, start1) / 1000.0 << "\n";
	auto start2 = getMoment;
	graphcut.push_relabel();
		sampleGpuMemUsage();
	auto end2 = getMoment;
	cout << "Kernel Time = " << TimeCpu(end2, start2) / 1000.0 << "\n";
	auto start3 = getMoment;
	graphcut.selectPix(result, result1);
		sampleGpuMemUsage();
	auto end3 = getMoment;
	cout << "Stitching Time = " << TimeCpu(end3, start3) / 1000.0 << "\n";
	auto end = getMoment;
	cout << "Total Time = " << TimeCpu(end, start) / 1000.0 << "\n";
	graphcut.cudaCutsFreeMem();
		sampleGpuMemUsage();
		extern void showGpuMemPeak();
		showGpuMemPeak();
	imshow("result", result);
	imshow("result1_stitching", result1);
	cv::waitKey(0);
	return 0;
}