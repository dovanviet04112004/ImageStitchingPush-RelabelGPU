#ifndef CUDACUT_H
#define CUDACUT_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include <fstream>
#include <chrono>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"

#define getMoment std::chrono::high_resolution_clock::now()
#define TimeCpu(end,start) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
using namespace std;
using namespace cv;

#define gpuErrChk(call) {gpuError((call));}
inline void gpuError(cudaError_t call) {
    const cudaError_t error = call;
    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", __FILE__, __LINE__);
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

class CudaCut
{
public:
    CudaCut();
    CudaCut(int image_width, int image_height, int overlap_width);

public:
    void h_mem_init();
    void d_mem_init();
    int cudaCutsInit();
    int cudaCutsSetupGraph(cv::Mat& image1, cv::Mat& image2);
    void cudaCutsFreeMem();
    void selectPix(cv::Mat& result, cv::Mat& result1);
    void global_relabel();
    void global_relabel_CPU();
    void push_relabel();

public:
    vector<int> vec1, vec2, vec3, vec4;
    int width, height, graph_size, size_int, image_width, image_width_B;
    cv::Mat area1, area2, process_are, result;
    dim3 grid, block;

    int* d_left_weight, * d_right_weight, * d_down_weight, * d_up_weight;
    int* d_excess_flow;
    int* d_relabel_mask;
    int* d_graph_height;
    int* d_height_backup;
    int* d_visited;
    bool* d_frontier;
    bool* d_next_frontier;
    int* d_changed;
    int* d_m1, * d_m2, * d_process_area, * d_horizontal, * d_vertical;
    int* d_push_block_position;
    int* d_up_right_sum, * d_up_left_sum;
    int* d_down_right_sum, * d_down_left_sum;

    int* h_left_weight, * h_right_weight, * h_down_weight, * h_up_weight;
    int* h_excess_flow;
    int* h_relabel_mask;
    int* h_graph_height;
    int* h_height_backup;
    int* h_visited;
    bool* h_frontier;
    unsigned char* h_m1, * h_m2;
    int* h_process_area, * h_horizontal, * h_vertical;
    int* h_push_block_position;
    int* h_up_right_sum, * h_up_left_sum;
    int* h_down_right_sum, * h_down_left_sum;
    int* h_active_node;

    bool* scanned, * mark;
    int* Excess_total;
    int* excess_source, * excess_sink;
    bool use_gpu_global_relabel;
    int* count_check;
};

// GPU kernel declarations
__global__ void computeOverlapDiff_kernel(const uchar* d_A, const uchar* d_B, 
    double* d_diff_results, int width_A, int img_height, int overlap, int width_B);
__global__ void reduceRowDiffs_kernel(double* d_diff_results, double* d_total_diff, int num_rows);
__global__ void selectPixGPU_kernel(uchar* d_result, const uchar* d_area1, const uchar* d_area2, 
    const bool* d_scanned, int result_width, int overlap_width, int overlap_height, int xoffset);
__global__ void findSeamPos_kernel(const bool* d_scanned, int* d_seam_pos, int overlap_width, int overlap_height);
__global__ void blendSeam_kernel(uchar* d_result, const uchar* d_area1, const uchar* d_area2, 
    const int* d_seam_pos, int result_width, int overlap_width, int overlap_height, 
    int xoffset, int blend_width);
__global__ void redistribute_excess_kernel(int* d_right_weight, int* d_left_weight, 
    int* d_up_weight, int* d_down_weight, int* d_excess_flow, int* d_graph_height, 
    int width, int height, int N);
__global__ void bfs_init_kernel(int* d_graph_height, int* d_visited, bool* d_frontier, 
    int width, int height, int N);
__global__ void bfs_step_kernel(int* d_right_weight, int* d_left_weight, int* d_up_weight, int* d_down_weight,
    int* d_graph_height, int* d_visited, bool* d_frontier, bool* d_next_frontier, 
    int width, int height, int N, int current_level, int* d_changed);
__global__ void swap_frontier_kernel(bool* d_frontier, bool* d_next_frontier, int N);
__global__ void processUnvisited_kernel(int* d_excess_flow, int* d_visited, int* d_Excess_total,
    int width, int height, int N);
__global__ void computeExcessSourceSink_kernel(int* d_excess_flow, int* excess_source, int* excess_sink,
    int width, int N);

// GPU Multi-band Blending
__global__ void blendLaplacianGPU_kernel(const short* lapA, const short* lapB, const float* mask,
    short* output, int width, int height, int channels);
__global__ void downsampleMaskGPU_kernel(const float* src, float* dst,
    int src_width, int src_height, int dst_width, int dst_height);
__global__ void addLaplacianGPU_kernel(const float* upsampled, const short* laplacian,
    float* output, int width, int height, int channels);
__global__ void upsampleGPU_kernel(const float* src, float* dst,
    int src_width, int src_height, int dst_width, int dst_height, int channels);

// GPU Multi-band blend function
cv::Mat multiBandBlendGPU(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Mat& mask, int levels);

#endif // CUDACUT_H