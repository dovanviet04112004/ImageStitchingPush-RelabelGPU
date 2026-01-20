#ifndef CUDACUT_H
#define CUDACUT_H

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include<opencv2/opencv.hpp>
#include <chrono>

#define WIDTH 1008
#define HEIGHT 755
#define OVERLAP_WIDTH 100

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

    // This function constructs the graph on the device
    int cudaCutsSetupGraph(cv::Mat& image1, cv::Mat& image2);


    // This function calls 3 kernels which performs the push, pull and relabel operation
    void cudaCutsAtomic(int blockDimy, int number_loops);

    // De-allocates all the memory allocated on the host and the device
    void cudaCutsFreeMem();
    void cudaWarmUp();
    // Functions calculates the total energy of the configuration
    void globalRelabelCpu(int* h_right_weight, int* h_left_weight, int* h_down_weight, int* h_up_weight, bool* visited, int* h_graph_height);
    void BfsCPU(int* h_right_weight, int* h_left_weight, int* h_down_weight, int* h_up_weight, bool* visited);
    int BfsCpuBackward(int* h_right_weight, int* h_left_weight, int* h_down_weight, int* h_up_weight, bool* visited);
    void getStitchingImage(cv::Mat& result, cv::Mat& result1);
    void selectPix(cv::Mat& result, cv::Mat& result1);
    void checkRelabelHight(int* d_relabel_mask, bool* visited);
    void global_relabel();
    void push_relabel();

public:
    /*************************************************
     * n-edges and t-edges                          **
     * **********************************************/
    vector<int> vec1, vec2, vec3, vec4;
    int width, height, graph_size, size_int, image_width;
    cv::Mat area1, area2, process_are, result;
    dim3 grid, block;

    int* d_left_weight, * d_right_weight, * d_down_weight, * d_up_weight;
    int* d_excess_flow;
    int* d_relabel_mask;
    int* d_graph_height;
    int* d_height_backup;
    int* d_visited; //for bfs
    bool* d_frontier; //for bfs
    int* d_m1, * d_m2, * d_process_area, * d_horizontal, * d_vertical;
    int* d_push_block_position;
    int* d_up_right_sum, * d_up_left_sum;
    int* d_down_right_sum, * d_down_left_sum;

    int* h_left_weight, * h_right_weight, * h_down_weight, * h_up_weight;
    int* h_excess_flow;
    int* h_relabel_mask;
    int* h_graph_height;
    int* h_height_backup;

    int* h_visited; // for bfs
    bool* h_frontier; // for bfs

    unsigned char* h_m1, * h_m2;
    int* h_process_area, * h_horizontal, * h_vertical;
    int* h_push_block_position;
    int* h_up_right_sum, * h_up_left_sum;
    int* h_down_right_sum, * h_down_left_sum;
    int* h_active_node;

    bool* scanned, * mark;

    int* Excess_total;

    int* excess_source, * excess_sink;
    int* count_check;

    //    int counter ;

};

#endif // CUDACUT_H
