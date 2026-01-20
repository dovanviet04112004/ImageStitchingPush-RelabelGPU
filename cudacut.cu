#include "cudacut.cuh"
#include "CudaCut_kernel.cu"

CudaCut::CudaCut(int image_width, int image_height, int overlap_width)
    : height(image_height), width(overlap_width), image_width(image_width), image_width_B(0), use_gpu_global_relabel(true) {
    graph_size = width * height;
    size_int = sizeof(int) * graph_size;


}


void CudaCut::h_mem_init()
{

    h_left_weight = (int*)malloc(size_int);
    h_right_weight = (int*)malloc(size_int);
    h_down_weight = (int*)malloc(size_int);
    h_up_weight = (int*)malloc(size_int);

    h_graph_height = (int*)malloc(size_int);

    h_excess_flow = (int*)malloc(size_int);
    h_relabel_mask = (int*)malloc(size_int);
    h_height_backup = (int*)malloc(4 * size_int);

    h_visited = (int*)malloc(size_int);
    h_frontier = (bool*)malloc(sizeof(bool) * graph_size);

    h_m1 = (unsigned char*)malloc(sizeof(unsigned char) * graph_size);
    h_m2 = (unsigned char*)malloc(sizeof(unsigned char) * graph_size);
    h_process_area = (int*)malloc(size_int);
    h_horizontal = (int*)malloc(size_int + height * sizeof(int));
    h_vertical = (int*)malloc(size_int + width * sizeof(int));


    h_push_block_position = (int*)malloc(sizeof(int) * (5 * height));
    h_up_right_sum = (int*)malloc(size_int);
    h_up_left_sum = (int*)malloc(size_int);
    h_down_right_sum = (int*)malloc(size_int);
    h_down_left_sum = (int*)malloc(size_int);
    h_active_node = (int*)malloc(sizeof(int) * 64);
    memset(h_active_node, 0, sizeof(int) * 64);

    scanned = (bool*)malloc(sizeof(bool) * graph_size);
    mark = (bool*)malloc(sizeof(bool) * graph_size);

    count_check = (int*)malloc(sizeof(int) * 64);
    *count_check = 2;


    // initial h_weight, h_flow from input

}

void CudaCut::d_mem_init()
{
  
    cudaMallocManaged((void**)&d_left_weight, size_int);
    cudaMallocManaged((void**)&d_right_weight, size_int);
    cudaMallocManaged((void**)&d_down_weight, size_int);
    cudaMallocManaged((void**)&d_up_weight, size_int);
    cudaMallocManaged((void**)&d_excess_flow, size_int);
    cudaMallocManaged((void**)&d_horizontal, size_int + height * sizeof(int));
    cudaMallocManaged((void**)&d_vertical, size_int + width * sizeof(int));
    cudaMallocManaged((void**)&d_graph_height, size_int);
    cudaMallocManaged((void**)&Excess_total, sizeof(int));
    *Excess_total = 0;
    cudaMallocManaged((void**)&excess_source, sizeof(int));
    *excess_source = 0;
    cudaMallocManaged((void**)&excess_sink, sizeof(int));
    *excess_sink = 0;


    //gpuErrChk(cudaMalloc((void**)&d_graph_height, size_int));
    //gpuErrChk(cudaMalloc((void**)&d_excess_flow, size_int));
    gpuErrChk(cudaMalloc((void**)&d_relabel_mask, size_int));

    gpuErrChk(cudaMalloc((void**)&d_height_backup, 4 * size_int));
    gpuErrChk(cudaMalloc((void**)&d_visited, size_int));
    gpuErrChk(cudaMalloc((void**)&d_frontier, sizeof(bool) * graph_size));
    gpuErrChk(cudaMalloc((void**)&d_next_frontier, sizeof(bool) * graph_size));
    gpuErrChk(cudaMalloc((void**)&d_changed, sizeof(int)));

    gpuErrChk(cudaMalloc((void**)&d_m1, size_int));
    gpuErrChk(cudaMalloc((void**)&d_m2, size_int));
    gpuErrChk(cudaMalloc((void**)&d_process_area, size_int));
    //gpuErrChk(cudaMalloc((void**)&d_horizontal, size_int + height*sizeof(int)));
    //gpuErrChk(cudaMalloc((void**)&d_vertical, size_int + width*sizeof(int)));


    gpuErrChk(cudaMalloc((void**)&d_push_block_position, sizeof(int) * (5 * height)));
    gpuErrChk(cudaMalloc((void**)&d_up_right_sum, size_int));
    gpuErrChk(cudaMalloc((void**)&d_up_left_sum, size_int));
    gpuErrChk(cudaMalloc((void**)&d_down_right_sum, size_int));
    gpuErrChk(cudaMalloc((void**)&d_down_left_sum, size_int));


}


int CudaCut::cudaCutsSetupGraph(cv::Mat& image1, cv::Mat& image2) {
    // Height must match
    if (image1.rows != height || image2.rows != height) {
        cout << "Height mismatch: expected " << height
             << ", got A=" << image1.rows << ", B=" << image2.rows << endl;
        return -1;
    }
    
    // Lưu chiều ngang của từng ảnh
    image_width = image1.cols;
    image_width_B = image2.cols;

    // Copy overlapped strips
    int xoffset = image_width - width;
    image1(cv::Rect(xoffset, 0, width, height)).copyTo(area1);
    image2(cv::Rect(0, 0, width, height)).copyTo(area2);
    // Initialize horizontal weights (contiguous per-row layout)
    for (int y = 0; y < height; y++) {
        const uchar* pA = image1.ptr<uchar>(y) + xoffset;
        const uchar* pB = image2.ptr<uchar>(y);
        int* pH = &d_horizontal[y * (width + 1)];
        
        pH[0] = 0;  // No edge before first column (source boundary)
        for (int x = 0; x < width - 1; x++) {
            int cap0 = abs(pA[x] - pB[x]);
            int cap1 = abs(pA[x + 1] - pB[x + 1]);
            pH[x + 1] = cap0 + cap1;
        }
        pH[width] = 1 << 20;  // Large capacity to sink
    }
    // Initialize vertical weights
    for (int x = 0; x < width; x++) {
        d_vertical[x] = 0;
        d_vertical[height * width + x] = 0;
    }
    for (int y = 0; y < height - 1; y++) {
        const uchar* pA0 = image1.ptr<uchar>(y) + xoffset;
        const uchar* pB0 = image2.ptr<uchar>(y);
        const uchar* pA1 = image1.ptr<uchar>(y + 1) + xoffset;
        const uchar* pB1 = image2.ptr<uchar>(y + 1);
        int* pV = &d_vertical[(y + 1) * width];
        
        for (int x = 0; x < width; x++) {
            int cap0 = abs(pA0[x] - pB0[x]);
            int cap1 = abs(pA1[x] - pB1[x]);
            pV[x] = cap0 + cap1;
        }
    }
    
    // Launch graph-setup kernels
    dim3 block(32, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    setupGraph_kernel<<<grid, block>>>(d_horizontal, d_vertical, d_right_weight, d_left_weight, d_up_weight,
        d_down_weight, d_excess_flow, d_push_block_position, d_graph_height, d_relabel_mask, width, height, graph_size);
    adjustGraph_kernel<<<grid, block>>>(d_excess_flow, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_graph_height,
        d_up_right_sum, d_up_left_sum, d_down_right_sum, d_down_left_sum, width, height, graph_size, Excess_total);
    
    cudaDeviceSynchronize();
    cout<<"setup graph done (GPU-accelerated)"<<endl;
    return 0;
}

int CudaCut::cudaCutsInit() {
    d_mem_init();
    h_mem_init();
    return 0;
}




void CudaCut::cudaCutsFreeMem()
{

    free(h_left_weight);
    free(h_right_weight);
    free(h_down_weight);
    free(h_up_weight);

    free(h_excess_flow);
    free(h_relabel_mask);
    free(h_graph_height);
    free(h_height_backup);
    free(h_visited);
    free(h_frontier);

    free(h_m1);
    free(h_m2);
    free(h_process_area);
    free(h_horizontal);
    free(h_vertical);

    free(h_push_block_position);
    free(h_up_right_sum);
    free(h_up_left_sum);
    free(h_down_right_sum);
    free(h_down_left_sum);
    free(scanned);
    free(mark);
    free(count_check);




    gpuErrChk(cudaFree(d_left_weight));
    gpuErrChk(cudaFree(d_right_weight));
    gpuErrChk(cudaFree(d_down_weight));
    gpuErrChk(cudaFree(d_up_weight));

    gpuErrChk(cudaFree(d_excess_flow));
    gpuErrChk(cudaFree(d_relabel_mask));
    gpuErrChk(cudaFree(d_graph_height));
    gpuErrChk(cudaFree(d_height_backup));
    gpuErrChk(cudaFree(d_visited));
    gpuErrChk(cudaFree(d_frontier));
    gpuErrChk(cudaFree(d_next_frontier));
    gpuErrChk(cudaFree(d_changed));

    gpuErrChk(cudaFree(d_m1));
    gpuErrChk(cudaFree(d_m2));
    gpuErrChk(cudaFree(d_process_area));
    gpuErrChk(cudaFree(d_horizontal));
    gpuErrChk(cudaFree(d_vertical));

    gpuErrChk(cudaFree(d_push_block_position));
    gpuErrChk(cudaFree(d_up_right_sum));
    gpuErrChk(cudaFree(d_up_left_sum));
    gpuErrChk(cudaFree(d_down_right_sum));
    gpuErrChk(cudaFree(Excess_total));
    gpuErrChk(cudaFree(excess_source));
    gpuErrChk(cudaFree(excess_sink));
}

void CudaCut::global_relabel_CPU() {
    // ===== ORIGINAL CPU BFS =====
    for (int node = 0; node < graph_size; node++) {
        int x = node % width;
        int y = node / width;
        
        if (y > 0 && y < height - 1 && x != 0 && (x + 1) != width && d_excess_flow[node] > 0) {
            vector<int> node_nei = { node + 1, node - 1, node + width, node - width };
            for (int nei : node_nei) {
                if (nei == node + 1 && d_graph_height[node] > d_graph_height[node + 1] && d_right_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_right_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node + 1] += e_flow;
                    d_right_weight[node] -= e_flow;
                    d_left_weight[node + 1] += e_flow;
                }
                else if (nei == node - 1 && d_graph_height[node] > d_graph_height[node - 1] && d_left_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_left_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node - 1] += e_flow;
                    d_left_weight[node] -= e_flow;
                    d_right_weight[node - 1] += e_flow;
                }
                else if (nei == node + width && d_graph_height[node] > d_graph_height[node + width] && d_down_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_down_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node + width] += e_flow;
                    d_down_weight[node] -= e_flow;
                    d_up_weight[node + width] += e_flow;
                }
                else if (nei == node - width && d_graph_height[node] > d_graph_height[node - width] && d_up_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_up_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node - width] += e_flow;
                    d_up_weight[node] -= e_flow;
                    d_down_weight[node - width] += e_flow;
                }
            }
        }
        else if (y == 0 && x != 0 && (x + 1) != width && d_excess_flow[node] > 0) {
            vector<int> node_nei = { node + 1, node - 1, node + width };
            for (int nei : node_nei) {
                if (nei == node + 1 && d_graph_height[node] > d_graph_height[node + 1] && d_right_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_right_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node + 1] += e_flow;
                    d_right_weight[node] -= e_flow;
                    d_left_weight[node + 1] += e_flow;
                }
                else if (nei == node - 1 && d_graph_height[node] > d_graph_height[node - 1] && d_left_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_left_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node - 1] += e_flow;
                    d_left_weight[node] -= e_flow;
                    d_right_weight[node - 1] += e_flow;
                }
                else if (nei == node + width && d_graph_height[node] > d_graph_height[node + width] && d_down_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_down_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node + width] += e_flow;
                    d_down_weight[node] -= e_flow;
                    d_up_weight[node + width] += e_flow;
                }
            }
        }
        else if (y == height - 1 && x != 0 && (x + 1) != width && d_excess_flow[node] > 0) {
            vector<int> node_nei = { node + 1, node - 1, node - width };
            for (int nei : node_nei) {
                if (nei == node + 1 && d_graph_height[node] > d_graph_height[node + 1] && d_right_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_right_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node + 1] += e_flow;
                    d_right_weight[node] -= e_flow;
                    d_left_weight[node + 1] += e_flow;
                }
                else if (nei == node - 1 && d_graph_height[node] > d_graph_height[node - 1] && d_left_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_left_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node - 1] += e_flow;
                    d_left_weight[node] -= e_flow;
                    d_right_weight[node - 1] += e_flow;
                }
                else if (nei == node - width && d_graph_height[node] > d_graph_height[node - width] && d_up_weight[node] > 0) {
                    int e_flow = min(d_excess_flow[node], d_up_weight[node]);
                    d_excess_flow[node] -= e_flow;
                    d_excess_flow[node - width] += e_flow;
                    d_up_weight[node] -= e_flow;
                    d_down_weight[node - width] += e_flow;
                }
            }
        }
        if (x == 0) {
            *excess_source += d_excess_flow[node];
        }
        if ((x + 1) == width) {
            *excess_sink += d_excess_flow[node];
        }
    }

    // BFS from sink
    memset(scanned, false, sizeof(bool) * graph_size);
    
    queue<int> que;
    int node1 = width - 1;
    for (int i = 0; i < height; i++) {
        que.push(node1 + i * width);
        scanned[node1 + i * width] = true;
        d_graph_height[node1 + i * width] = 0;
    }

    while (!que.empty()) {
        int s = que.front();
        que.pop();
        int current = d_graph_height[s] + 1;
        int x = s % width;
        int y = s / width;
        
        if (x != 0 && d_right_weight[s - 1] > 0 && !scanned[s - 1]) {
            d_graph_height[s - 1] = current;
            scanned[s - 1] = true;
            que.push(s - 1);
        }
        if (y < height - 1 && d_up_weight[s + width] > 0 && !scanned[s + width]) {
            d_graph_height[s + width] = current;
            scanned[s + width] = true;
            que.push(s + width);
        }
        if (y >= 1 && d_down_weight[s - width] > 0 && !scanned[s - width]) {
            d_graph_height[s - width] = current;
            scanned[s - width] = true;
            que.push(s - width);
        }
        if (x < width - 1 && d_left_weight[s + 1] > 0 && !scanned[s + 1]) {
            d_graph_height[s + 1] = current;
            scanned[s + 1] = true;
            que.push(s + 1);
        }
    }

    // Process unvisited nodes
    for (int i = 0; i < graph_size; i++) {
        if (!scanned[i] && i % width != 0 && (i + 1) % width != 0) {
            *Excess_total -= d_excess_flow[i];
            d_excess_flow[i] = 0;
        }
    }
}


// GPU PARALLEL BFS GLOBAL_RELABEL (COMMENTED OUT - CPU faster for narrow graphs)
void CudaCut::global_relabel() {
    int threadsPerBlock = 256;
    int numBlocks = (graph_size + threadsPerBlock - 1) / threadsPerBlock;

    
    redistribute_excess_kernel<<<numBlocks, threadsPerBlock>>>(
        d_right_weight, d_left_weight, d_up_weight, d_down_weight,
        d_excess_flow, d_graph_height, width, height, graph_size);
    gpuErrChk(cudaDeviceSynchronize());

    
    *excess_source = 0;
    *excess_sink = 0;
    computeExcessSourceSink_kernel<<<numBlocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(int)>>>(
        d_excess_flow, excess_source, excess_sink, width, graph_size);
    gpuErrChk(cudaDeviceSynchronize());

   
    bfs_init_kernel<<<numBlocks, threadsPerBlock>>>(
        d_graph_height, d_visited, d_frontier, width, height, graph_size);
    gpuErrChk(cudaDeviceSynchronize());

 
    int h_changed = 1;
    int current_level = 0;

    while (h_changed) {
        h_changed = 0;
        gpuErrChk(cudaMemcpy(d_changed, &h_changed, sizeof(int), cudaMemcpyHostToDevice));

        bfs_step_kernel<<<numBlocks, threadsPerBlock>>>(
            d_right_weight, d_left_weight, d_up_weight, d_down_weight,
            d_graph_height, d_visited, d_frontier, d_next_frontier,
            width, height, graph_size, current_level, d_changed);
        gpuErrChk(cudaDeviceSynchronize());

        swap_frontier_kernel<<<numBlocks, threadsPerBlock>>>(
            d_frontier, d_next_frontier, graph_size);
        gpuErrChk(cudaDeviceSynchronize());

        gpuErrChk(cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
        current_level++;
    }

    
    processUnvisited_kernel<<<numBlocks, threadsPerBlock>>>(
        d_excess_flow, d_visited, Excess_total, width, height, graph_size);
    gpuErrChk(cudaDeviceSynchronize());

 
    gpuErrChk(cudaMemcpy(h_visited, d_visited, size_int, cudaMemcpyDeviceToHost));
    for (int i = 0; i < graph_size; i++) {
        scanned[i] = (h_visited[i] != 0);
    }
}

void CudaCut::push_relabel()
{
    memset(mark, false, sizeof(bool) * graph_size);
    
    dim3 block(32, 16, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    
    int max_iterations = 100;
    int max_batches = 10;
    int iter = 0;
    int batch = 0;
    
    while (batch < max_batches) {
        
        for (int i = 0; i < max_iterations; i++) {
            push_relabel_kernel<<<grid, block>>>(d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                d_excess_flow, d_graph_height, d_relabel_mask, d_height_backup,
                width, height, graph_size, excess_source, excess_sink);
        }
        cudaDeviceSynchronize();
        iter += max_iterations;
        
        *excess_source = 0;
        *excess_sink = 0;
        if (use_gpu_global_relabel) {
            global_relabel();
        } else {
            global_relabel_CPU();
        }
        
        if ((*excess_source + *excess_sink) >= (*Excess_total)) {
            break;
        }
        batch++;
    }

    cout << "max_flow: " << *excess_sink << " (iterations: " << iter << ")" << endl;
}

// GPU-ACCELERATED PIXEL SELECTION 
void CudaCut::selectPix(cv::Mat& result, cv::Mat& result1) {
    int xoffset = image_width - width;
    
    // Allocate GPU memory
    uchar *d_result, *d_area1, *d_area2;
    bool *d_scanned;
    int *d_seam_pos;
    
    int result_size = result.cols * result.rows;
    cudaMalloc(&d_result, result_size);
    cudaMalloc(&d_area1, graph_size);
    cudaMalloc(&d_area2, graph_size);
    cudaMalloc(&d_scanned, sizeof(bool) * graph_size);
    cudaMalloc(&d_seam_pos, sizeof(int) * height);
    
    // Copy data to GPU
    cudaMemcpy(d_result, result.data, result_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_area1, area1.data, graph_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_area2, area2.data, graph_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scanned, scanned, sizeof(bool) * graph_size, cudaMemcpyHostToDevice);
    
    dim3 block2d(32, 16);
    dim3 grid2d((width + block2d.x - 1) / block2d.x, (height + block2d.y - 1) / block2d.y);
    
    // Select pixels on GPU
    selectPixGPU_kernel<<<grid2d, block2d>>>(d_result, d_area1, d_area2, d_scanned, 
        result.cols, width, height, xoffset);
    
    // Find seam positions
    int threads = 256;
    int blocks = (height + threads - 1) / threads;
    findSeamPos_kernel<<<blocks, threads>>>(d_scanned, d_seam_pos, width, height);
    
    // Blend along seam
    int blend_width = max(5, width / 40);
    blendSeam_kernel<<<grid2d, block2d>>>(d_result, d_area1, d_area2, d_seam_pos, 
        result.cols, width, height, xoffset, blend_width);
    
    cudaDeviceSynchronize();
    
    // Copy result back
    cudaMemcpy(result.data, d_result, result_size, cudaMemcpyDeviceToHost);
    
    // Create result1 with seam highlighted
    result.copyTo(result1);
    int* h_seam_pos = new int[height];
    cudaMemcpy(h_seam_pos, d_seam_pos, sizeof(int) * height, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < height; i++) {
        if (h_seam_pos[i] > 0) {
            result1.at<uchar>(i, xoffset + h_seam_pos[i]) = 255;
            result1.at<uchar>(i, xoffset + h_seam_pos[i] + 1) = 255;
        }
    }
    
    // Cleanup
    delete[] h_seam_pos;
    cudaFree(d_result);
    cudaFree(d_area1);
    cudaFree(d_area2);
    cudaFree(d_scanned);
    cudaFree(d_seam_pos);
}

// GPU-ACCELERATED Auto-detect overlap

int autoDetectOverlapGPU(const cv::Mat& A, const cv::Mat& B) {
    int width_A = A.cols;
    int width_B = B.cols;
    int height = A.rows;
    
    
    int min_width = min(width_A, width_B);
    int min_overlap = max(20, min_width / 20);  
    int max_overlap = (int)(min_width * 0.9);    
    int step = 5;  
    
    
    uchar *d_A, *d_B;
    double *d_diff_results, *d_total_diff;
    
    int img_size_A = width_A * height;
    int img_size_B = width_B * height;
    cudaMalloc(&d_A, img_size_A);
    cudaMalloc(&d_B, img_size_B);
    cudaMalloc(&d_diff_results, sizeof(double) * height);
    cudaMallocManaged(&d_total_diff, sizeof(double));
    
    cudaMemcpy(d_A, A.data, img_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data, img_size_B, cudaMemcpyHostToDevice);
    
    int threads = 256;
    int shared_size = threads * sizeof(double);
    
    int best_overlap = min_overlap;
    double min_diff = DBL_MAX;
    
    
    for (int overlap = min_overlap; overlap <= max_overlap; overlap += step) {
        *d_total_diff = 0;
        
        
        computeOverlapDiff_kernel<<<height, threads, shared_size>>>(
            d_A, d_B, d_diff_results, width_A, height, overlap, width_B);
        
       
        reduceRowDiffs_kernel<<<1, threads, shared_size>>>(d_diff_results, d_total_diff, height);
        cudaDeviceSynchronize();
        
        double avg_diff = *d_total_diff / (height * overlap);
        if (avg_diff < min_diff) {
            min_diff = avg_diff;
            best_overlap = overlap;
        }
    }
    
    // Fine search với range rộng hơn (±20 pixels)
    int refined_overlap = best_overlap;
    min_diff = DBL_MAX;
    
    for (int overlap = max(min_overlap, best_overlap - 20); 
         overlap <= min(max_overlap, best_overlap + 20); overlap++) {
        *d_total_diff = 0;
        
        computeOverlapDiff_kernel<<<height, threads, shared_size>>>(
            d_A, d_B, d_diff_results, width_A, height, overlap, width_B);
        reduceRowDiffs_kernel<<<1, threads, shared_size>>>(d_diff_results, d_total_diff, height);
        cudaDeviceSynchronize();
        
        double avg_diff = *d_total_diff / (height * overlap);
        if (avg_diff < min_diff) {
            min_diff = avg_diff;
            refined_overlap = overlap;
        }
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_diff_results);
    cudaFree(d_total_diff);
    
    return (refined_overlap / 2) * 2;
}


// GPU Multi-band Blending

// Build Gaussian pyramid của mask trên GPU
void buildGaussianPyramidMaskGPU(const cv::Mat& mask, std::vector<float*>& d_mask_pyramid, 
                                  std::vector<int>& widths, std::vector<int>& heights, int levels) {
    // Level 0: convert mask sang float và copy lên GPU
    cv::Mat mask_float;
    mask.convertTo(mask_float, CV_32F, 1.0/255.0);
    
    widths.push_back(mask.cols);
    heights.push_back(mask.rows);
    
    float* d_mask0;
    size_t size0 = mask.cols * mask.rows * sizeof(float);
    cudaMalloc(&d_mask0, size0);
    cudaMemcpy(d_mask0, mask_float.ptr<float>(), size0, cudaMemcpyHostToDevice);
    d_mask_pyramid.push_back(d_mask0);
    
    // Build pyramid bằng downsample kernel
    dim3 block(16, 16);
    
    for (int i = 1; i < levels; i++) {
        int prev_w = widths[i-1];
        int prev_h = heights[i-1];
        int cur_w = (prev_w + 1) / 2;
        int cur_h = (prev_h + 1) / 2;
        
        widths.push_back(cur_w);
        heights.push_back(cur_h);
        
        float* d_mask_cur;
        cudaMalloc(&d_mask_cur, cur_w * cur_h * sizeof(float));
        
        dim3 grid((cur_w + 15) / 16, (cur_h + 15) / 16);
        downsampleMaskGPU_kernel<<<grid, block>>>(
            d_mask_pyramid[i-1], d_mask_cur,
            prev_w, prev_h, cur_w, cur_h
        );
        
        d_mask_pyramid.push_back(d_mask_cur);
    }
    
    cudaDeviceSynchronize();
}

// Blend 2 Laplacian pyramids trên GPU
void blendLaplacianPyramidsGPU(
    const std::vector<cv::Mat>& lapA,
    const std::vector<cv::Mat>& lapB,
    const std::vector<float*>& d_mask_pyramid,
    const std::vector<int>& widths,
    const std::vector<int>& heights,
    std::vector<cv::Mat>& blended,
    int channels
) {
    int levels = (int)lapA.size();
    blended.resize(levels);
    
    dim3 block(16, 16);
    
    for (int i = 0; i < levels; i++) {
        int w = widths[i];
        int h = heights[i];
        size_t lap_size = w * h * channels * sizeof(short);
        
      
        short *d_lapA, *d_lapB, *d_output;
        cudaMalloc(&d_lapA, lap_size);
        cudaMalloc(&d_lapB, lap_size);
        cudaMalloc(&d_output, lap_size);
        
        
        cudaMemcpy(d_lapA, lapA[i].ptr<short>(), lap_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lapB, lapB[i].ptr<short>(), lap_size, cudaMemcpyHostToDevice);
        
        
        dim3 grid((w + 15) / 16, (h + 15) / 16);
        blendLaplacianGPU_kernel<<<grid, block>>>(
            d_lapA, d_lapB, d_mask_pyramid[i], d_output,
            w, h, channels
        );
        
       
        blended[i] = cv::Mat(h, w, (channels == 3) ? CV_16SC3 : CV_16SC1);
        cudaMemcpy(blended[i].ptr<short>(), d_output, lap_size, cudaMemcpyDeviceToHost);
        
       
        cudaFree(d_lapA);
        cudaFree(d_lapB);
        cudaFree(d_output);
    }
    
    cudaDeviceSynchronize();
}

// Reconstruct từ Laplacian pyramid trên GPU
cv::Mat reconstructFromLaplacianGPU(const std::vector<cv::Mat>& lap_pyr, int channels) {
    int levels = (int)lap_pyr.size();
    
    
    cv::Mat current;
    lap_pyr[levels - 1].convertTo(current, CV_32F);
    
    dim3 block(16, 16);
    
    
    for (int i = levels - 2; i >= 0; i--) {
        int target_w = lap_pyr[i].cols;
        int target_h = lap_pyr[i].rows;
        int src_w = current.cols;
        int src_h = current.rows;
        
        size_t src_size = src_w * src_h * channels * sizeof(float);
        size_t dst_size = target_w * target_h * channels * sizeof(float);
        size_t lap_size = target_w * target_h * channels * sizeof(short);
        
       
        float *d_src, *d_upsampled, *d_output;
        short *d_lap;
        cudaMalloc(&d_src, src_size);
        cudaMalloc(&d_upsampled, dst_size);
        cudaMalloc(&d_output, dst_size);
        cudaMalloc(&d_lap, lap_size);
        
        
        cudaMemcpy(d_src, current.ptr<float>(), src_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_lap, lap_pyr[i].ptr<short>(), lap_size, cudaMemcpyHostToDevice);
        
        
        dim3 grid_up((target_w + 15) / 16, (target_h + 15) / 16);
        upsampleGPU_kernel<<<grid_up, block>>>(
            d_src, d_upsampled,
            src_w, src_h, target_w, target_h, channels
        );
        
        
        addLaplacianGPU_kernel<<<grid_up, block>>>(
            d_upsampled, d_lap, d_output,
            target_w, target_h, channels
        );
        

        current = cv::Mat(target_h, target_w, (channels == 3) ? CV_32FC3 : CV_32FC1);
        cudaMemcpy(current.ptr<float>(), d_output, dst_size, cudaMemcpyDeviceToHost);
        
       
        cudaFree(d_src);
        cudaFree(d_upsampled);
        cudaFree(d_output);
        cudaFree(d_lap);
    }
    
    cudaDeviceSynchronize();
    
    cv::Mat result;
    current.convertTo(result, (channels == 3) ? CV_8UC3 : CV_8UC1);
    return result;
}

// Main GPU Multi-band blend function
cv::Mat multiBandBlendGPU(const cv::Mat& imgA, const cv::Mat& imgB, const cv::Mat& mask, int levels) {
    int channels = imgA.channels();
    int cv_type_16s = (channels == 3) ? CV_16SC3 : CV_16SC1;
    
    std::vector<float*> d_mask_pyramid;
    std::vector<int> widths, heights;
    buildGaussianPyramidMaskGPU(mask, d_mask_pyramid, widths, heights, levels);
    
    std::vector<cv::Mat> gaussA, gaussB;
    gaussA.push_back(imgA.clone());
    gaussB.push_back(imgB.clone());
    
    for (int i = 0; i < levels - 1; i++) {
        cv::Mat downA, downB;
        cv::pyrDown(gaussA[i], downA);
        cv::pyrDown(gaussB[i], downB);
        gaussA.push_back(downA);
        gaussB.push_back(downB);
    }
    
    std::vector<cv::Mat> lapA(levels), lapB(levels);
    for (int i = 0; i < levels - 1; i++) {
        cv::Mat upA, upB;
        cv::pyrUp(gaussA[i + 1], upA, gaussA[i].size());
        cv::pyrUp(gaussB[i + 1], upB, gaussB[i].size());
        
        cv::Mat diffA, diffB;
        cv::subtract(gaussA[i], upA, diffA, cv::noArray(), cv_type_16s);
        cv::subtract(gaussB[i], upB, diffB, cv::noArray(), cv_type_16s);
        lapA[i] = diffA;
        lapB[i] = diffB;
    }

    gaussA[levels - 1].convertTo(lapA[levels - 1], cv_type_16s);
    gaussB[levels - 1].convertTo(lapB[levels - 1], cv_type_16s);
    
    std::vector<cv::Mat> blended;
    blendLaplacianPyramidsGPU(lapA, lapB, d_mask_pyramid, widths, heights, blended, channels);
    
    cv::Mat result = reconstructFromLaplacianGPU(blended, channels);
    
    for (auto& d_mask : d_mask_pyramid) {
        cudaFree(d_mask);
    }
    
    return result;
}
