#include "cudacut.cuh"

// GPU kernel: setup graph structure
__global__ void
setupGraph_kernel(int* d_horizontal, int* d_vertical, int* d_right_weight, int* d_left_weight, int* d_up_weight,
    int* d_down_weight, int* d_excess_flow, int* d_push_block_position, int* d_graph_height, int* d_relabel_mask,
    int width, int height, int N) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (ix >= width || iy >= height) return;
    
    int node_i = iy * width + ix;
    
    // d_horizontal has (width+1) elements per row
    d_right_weight[node_i] = d_horizontal[iy * (width + 1) + ix + 1];
    d_left_weight[node_i] = d_horizontal[iy * (width + 1) + ix];

    d_down_weight[node_i] = d_vertical[(iy + 1) * width + ix];
    d_up_weight[node_i] = d_vertical[node_i];
    d_excess_flow[node_i] = 0;
    d_graph_height[node_i] = width - ix - 1;
    d_relabel_mask[node_i] = 0;
}

__global__ void
adjustGraph_kernel(int* d_excess_flow, int* d_left_weight, int* d_right_weight, int* d_down_weight, int* d_up_weight,
    int* d_graph_height, int* d_up_right_sum, int* d_up_left_sum, int* d_down_right_sum, int* d_down_left_sum,
    int width, int height, int N, int* Excess_total) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int node_i = iy * width + ix;

    if (ix == 0) {
        d_right_weight[node_i] = 0;
        d_up_weight[node_i] = 0;
        d_down_weight[node_i] = 0;
        d_graph_height[node_i] = N;
    }
    if (ix == width - 1) {
        d_left_weight[node_i] = 0;
        d_up_weight[node_i] = 0;
        d_down_weight[node_i] = 0;
    }

    if (ix == 1) {
        int tmp = d_left_weight[node_i];
        atomicAdd(Excess_total, tmp);
        d_excess_flow[node_i] = tmp;
        d_left_weight[node_i] = tmp + tmp;
    }
}

// GPU kernel: auto-detect overlap (parallel reduction)

__global__ void computeOverlapDiff_kernel(const uchar* d_A, const uchar* d_B, 
    double* d_diff_results, int width_A, int img_height, int overlap, int width_B) {
    
    extern __shared__ double shared_diff[];
    
    int tid = threadIdx.x;
    int row = blockIdx.x;  // Mỗi block xử lý 1 hàng
    
    if (row >= img_height) return;
    
    
    double local_diff = 0;
    for (int j = tid; j < overlap; j += blockDim.x) {
        int idxA = row * width_A + (width_A - overlap) + j;
        int idxB = row * width_B + j;
        local_diff += abs((double)d_A[idxA] - (double)d_B[idxB]);
    }
    
    shared_diff[tid] = local_diff;
    __syncthreads();
    
    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_diff[tid] += shared_diff[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        d_diff_results[row] = shared_diff[0];
    }
}

// Reduce per-row diffs to total
__global__ void reduceRowDiffs_kernel(double* d_diff_results, double* d_total_diff, int num_rows) {
    extern __shared__ double shared_sum[];
    
    int tid = threadIdx.x;
    double local_sum = 0;
    
    for (int i = tid; i < num_rows; i += blockDim.x) {
        local_sum += d_diff_results[i];
    }
    
    shared_sum[tid] = local_sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        *d_total_diff = shared_sum[0];
    }
}

// GPU kernel: select pixel (parallel stitching)
__global__ void selectPixGPU_kernel(uchar* d_result, const uchar* d_area1, const uchar* d_area2, 
    const bool* d_scanned, int result_width, int overlap_width, int overlap_height, int xoffset) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= overlap_width - 1 || y >= overlap_height) return;
    
    int scan_idx = y * overlap_width + x;
    int result_idx = y * result_width + xoffset + x;
    
    d_result[result_idx] = d_scanned[scan_idx] ? d_area2[scan_idx] : d_area1[scan_idx];
}

// Kernel: find seam position per row
__global__ void findSeamPos_kernel(const bool* d_scanned, int* d_seam_pos, int overlap_width, int overlap_height) {
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (y >= overlap_height) return;
    
    d_seam_pos[y] = -1;
    for (int x = 1; x < overlap_width - 1; x++) {
        int idx = y * overlap_width + x;
        if (d_scanned[idx] != d_scanned[idx + 1]) {
            d_seam_pos[y] = x;
            break;
        }
    }
}

// Kernel: blend along seam
__global__ void blendSeam_kernel(uchar* d_result, const uchar* d_area1, const uchar* d_area2, 
    const int* d_seam_pos, int result_width, int overlap_width, int overlap_height, 
    int xoffset, int blend_width) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y >= overlap_height) return;
    
    int seam_pos = d_seam_pos[y];
    if (seam_pos <= 0) return;
    
    int jStart = max(1, seam_pos - blend_width);
    int jEnd = min(overlap_width - 1, seam_pos + blend_width);
    
    if (x < jStart || x >= jEnd) return;
    
    float inv_blend = 1.0f / (2.0f * blend_width);
    float alpha = (float)(x - jStart) * inv_blend;
    alpha = fmaxf(0.0f, fminf(1.0f, alpha));
    
    int scan_idx = y * overlap_width + x;
    int result_idx = y * result_width + xoffset + x;
    
    d_result[result_idx] = (uchar)((1.0f - alpha) * d_area1[scan_idx] + alpha * d_area2[scan_idx]);
}

// GPU kernel: redistribute excess flow (part of push-relabel)
__global__ void redistribute_excess_kernel(
    int* d_right_weight, int* d_left_weight, int* d_up_weight, int* d_down_weight,
    int* d_excess_flow, int* d_graph_height, int width, int height, int N) {
    
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node >= N) return;
    
    int x = node % width;
    int y = node / width;
    int excess = d_excess_flow[node];
    
    if (excess <= 0) return;
    if (x == 0 || x == width - 1) return; 
    
    int my_height = d_graph_height[node];
    
    
    if (x < width - 1 && d_right_weight[node] > 0 && my_height > d_graph_height[node + 1]) {
        int flow = min(excess, d_right_weight[node]);
        atomicSub(&d_excess_flow[node], flow);
        atomicAdd(&d_excess_flow[node + 1], flow);
        atomicSub(&d_right_weight[node], flow);
        atomicAdd(&d_left_weight[node + 1], flow);
        excess -= flow;
    }
    
    if (excess <= 0) return;
    
    if (x > 0 && d_left_weight[node] > 0 && my_height > d_graph_height[node - 1]) {
        int flow = min(excess, d_left_weight[node]);
        atomicSub(&d_excess_flow[node], flow);
        atomicAdd(&d_excess_flow[node - 1], flow);
        atomicSub(&d_left_weight[node], flow);
        atomicAdd(&d_right_weight[node - 1], flow);
        excess -= flow;
    }
    
    if (excess <= 0) return;
    
    
    if (y < height - 1 && d_down_weight[node] > 0 && my_height > d_graph_height[node + width]) {
        int flow = min(excess, d_down_weight[node]);
        atomicSub(&d_excess_flow[node], flow);
        atomicAdd(&d_excess_flow[node + width], flow);
        atomicSub(&d_down_weight[node], flow);
        atomicAdd(&d_up_weight[node + width], flow);
        excess -= flow;
    }
    
    if (excess <= 0) return;
    
   
    if (y > 0 && d_up_weight[node] > 0 && my_height > d_graph_height[node - width]) {
        int flow = min(excess, d_up_weight[node]);
        atomicSub(&d_excess_flow[node], flow);
        atomicAdd(&d_excess_flow[node - width], flow);
        atomicSub(&d_up_weight[node], flow);
        atomicAdd(&d_down_weight[node - width], flow);
    }
}


// GPU Kernel: Parallel BFS for global_relabel
__global__ void bfs_init_kernel(int* d_graph_height, int* d_visited, bool* d_frontier, 
    int width, int height, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    d_visited[idx] = 0;
    d_frontier[idx] = false;
    
    // Khởi tạo frontier từ cột cuối (sink)
    int x = idx % width;
    if (x == width - 1) {
        d_frontier[idx] = true;
        d_visited[idx] = 1;
        d_graph_height[idx] = 0;
    }
}

// Optimized BFS step với atomic để tránh race conditions
__global__ void bfs_step_kernel(int* d_right_weight, int* d_left_weight, int* d_up_weight, int* d_down_weight,
    int* d_graph_height, int* d_visited, bool* d_frontier, bool* d_next_frontier, 
    int width, int height, int N, int current_level, int* d_changed) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    if (!d_frontier[idx]) return;
    
    int x = idx % width;
    int y = idx / width;
    int new_height = current_level + 1;
    
    // Left neighbor (x-1): dùng d_right_weight[idx-1]
    if (x > 0) {
        int neighbor = idx - 1;
        if (d_right_weight[neighbor] > 0 && atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
            d_next_frontier[neighbor] = true;
            d_graph_height[neighbor] = new_height;
            atomicExch(d_changed, 1);
        }
    }
    
    // Right neighbor (x+1): dùng d_left_weight[idx+1]
    if (x < width - 1) {
        int neighbor = idx + 1;
        if (d_left_weight[neighbor] > 0 && atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
            d_next_frontier[neighbor] = true;
            d_graph_height[neighbor] = new_height;
            atomicExch(d_changed, 1);
        }
    }
    
    // Up neighbor (y-1): dùng d_down_weight[idx-width]
    if (y > 0) {
        int neighbor = idx - width;
        if (d_down_weight[neighbor] > 0 && atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
            d_next_frontier[neighbor] = true;
            d_graph_height[neighbor] = new_height;
            atomicExch(d_changed, 1);
        }
    }
    
    // Down neighbor (y+1): dùng d_up_weight[idx+width]
    if (y < height - 1) {
        int neighbor = idx + width;
        if (d_up_weight[neighbor] > 0 && atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
            d_next_frontier[neighbor] = true;
            d_graph_height[neighbor] = new_height;
            atomicExch(d_changed, 1);
        }
    }
}

__global__ void swap_frontier_kernel(bool* d_frontier, bool* d_next_frontier, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    d_frontier[idx] = d_next_frontier[idx];
    d_next_frontier[idx] = false;
}

// Kernel xử lý excess flow cho các node chưa được visit
__global__ void processUnvisited_kernel(int* d_excess_flow, int* d_visited, int* d_Excess_total,
    int width, int height, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    
    int x = idx % width;
    if (d_visited[idx] == 0 && x != 0 && x != width - 1) {
        atomicSub(d_Excess_total, d_excess_flow[idx]);
        d_excess_flow[idx] = 0;
    }
}

// Kernel tính excess cho source/sink
__global__ void computeExcessSourceSink_kernel(int* d_excess_flow, int* excess_source, int* excess_sink,
    int width, int N) {
    extern __shared__ int shared_excess[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int local_source = 0;
    int local_sink = 0;
    
    if (idx < N) {
        int x = idx % width;
        if (x == 0) {
            local_source = d_excess_flow[idx];
        } else if (x == width - 1) {
            local_sink = d_excess_flow[idx];
        }
    }
    
    // Shared memory: [0..blockDim-1] = source, [blockDim..2*blockDim-1] = sink
    shared_excess[tid] = local_source;
    shared_excess[tid + blockDim.x] = local_sink;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_excess[tid] += shared_excess[tid + s];
            shared_excess[tid + blockDim.x] += shared_excess[tid + s + blockDim.x];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(excess_source, shared_excess[0]);
        atomicAdd(excess_sink, shared_excess[blockDim.x]);
    }
}

// push kernel
inline __global__ void
push_relabel_kernel(int* d_right_weight, int* d_left_weight, int* d_up_weight, int* d_down_weight,
    int* d_excess_flow, int* d_graph_height, int* d_relabel_mask, int* d_height_backup,
    int width, int height, int N, int* excess_source, int* excess_sink) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int node_i = iy * width + ix;

    if (node_i < N && ix != 0 && (ix + 1) != width) {
        int cycle = 1024;  // Mỗi thread xử lý 1024 push/relabel ops
        int h_dash, v_dash, d;

        while (cycle > 0) {
            if (d_excess_flow[node_i] > 0 && d_graph_height[node_i] < N) {
                h_dash = 1 << 22;
                v_dash = -1;

                if (iy > 0 && iy < height - 1) {
                    // Check all 4 neighbors with capacity > 0
                    if (d_left_weight[node_i] > 0 && d_graph_height[node_i - 1] < h_dash) {
                        h_dash = d_graph_height[node_i - 1];
                        v_dash = node_i - 1;
                    }
                    if (d_right_weight[node_i] > 0 && d_graph_height[node_i + 1] < h_dash) {
                        h_dash = d_graph_height[node_i + 1];
                        v_dash = node_i + 1;
                    }
                    if (d_up_weight[node_i] > 0 && d_graph_height[node_i - width] < h_dash) {
                        h_dash = d_graph_height[node_i - width];
                        v_dash = node_i - width;
                    }
                    if (d_down_weight[node_i] > 0 && d_graph_height[node_i + width] < h_dash) {
                        h_dash = d_graph_height[node_i + width];
                        v_dash = node_i + width;
                    }
                }
                else if (iy == 0) {
                    // First row - no up neighbor
                    if (d_left_weight[node_i] > 0 && d_graph_height[node_i - 1] < h_dash) {
                        h_dash = d_graph_height[node_i - 1];
                        v_dash = node_i - 1;
                    }
                    if (d_right_weight[node_i] > 0 && d_graph_height[node_i + 1] < h_dash) {
                        h_dash = d_graph_height[node_i + 1];
                        v_dash = node_i + 1;
                    }
                    if (d_down_weight[node_i] > 0 && d_graph_height[node_i + width] < h_dash) {
                        h_dash = d_graph_height[node_i + width];
                        v_dash = node_i + width;
                    }
                }
                else if (iy == height - 1) {
                    // Last row - no down neighbor
                    if (d_left_weight[node_i] > 0 && d_graph_height[node_i - 1] < h_dash) {
                        h_dash = d_graph_height[node_i - 1];
                        v_dash = node_i - 1;
                    }
                    if (d_right_weight[node_i] > 0 && d_graph_height[node_i + 1] < h_dash) {
                        h_dash = d_graph_height[node_i + 1];
                        v_dash = node_i + 1;
                    }
                    if (d_up_weight[node_i] > 0 && d_graph_height[node_i - width] < h_dash) {
                        h_dash = d_graph_height[node_i - width];
                        v_dash = node_i - width;
                    }
                }

                if (d_graph_height[node_i] > h_dash) {
                    
                    if (v_dash == node_i + 1) {
                        //printf("node_i_push: %d, v_dash: %d, excess_flow: %d, right_weight: %d\n", node_i, v_dash, d_excess_flow[node_i], d_right_weight[node_i]);
                        d = min(d_right_weight[node_i], d_excess_flow[node_i]);
                        atomicSub(&d_excess_flow[node_i], d);
                        atomicAdd(&d_excess_flow[v_dash], d);
                        atomicSub(&d_right_weight[node_i], d);
                        atomicAdd(&d_left_weight[v_dash], d);
                    }
                    if (v_dash == node_i + width) {
                        //printf("node_i_push: %d, v_dash: %d, excess_flow: %d, down_weight: %d\n", node_i, v_dash, d_excess_flow[node_i], d_down_weight[node_i]);
                        d = min(d_down_weight[node_i], d_excess_flow[node_i]);
                        atomicSub(&d_excess_flow[node_i], d);
                        atomicAdd(&d_excess_flow[v_dash], d);
                        atomicSub(&d_down_weight[node_i], d);
                        atomicAdd(&d_up_weight[v_dash], d);
                    }
                    if (v_dash == node_i - 1) {
                        //printf("node_i_push: %d, v_dash: %d, excess_flow: %d, left_weight: %d\n", node_i, v_dash, d_excess_flow[node_i], d_left_weight[node_i]);
                        d = min(d_left_weight[node_i], d_excess_flow[node_i]);
                        atomicSub(&d_excess_flow[node_i], d);
                        atomicAdd(&d_excess_flow[v_dash], d);
                        atomicSub(&d_left_weight[node_i], d);
                        atomicAdd(&d_right_weight[v_dash], d);
                    }
                    if (v_dash == node_i - width) {
                        //printf("node_i_push: %d, v_dash: %d, excess_flow: %d, up_weight: %d\n", node_i, v_dash, d_excess_flow[node_i], d_up_weight[node_i]);
                        d = min(d_up_weight[node_i], d_excess_flow[node_i]);
                        atomicSub(&d_excess_flow[node_i], d);
                        atomicAdd(&d_excess_flow[v_dash], d);
                        atomicSub(&d_up_weight[node_i], d);
                        atomicAdd(&d_down_weight[v_dash], d);
                    }
                }
                else if(v_dash >= 0 && v_dash < N){
                    d_graph_height[node_i] = h_dash + 1;
                    //printf("node_i_relabel: %d, d_graph_height: %d, h_dash: %d, excess_flow: %d, v_dash: %d\n", node_i, d_graph_height[node_i], h_dash, d_excess_flow[node_i], v_dash);
                }
                //__syncthreads();
                
                //printf("source: %d, sink: %d\n", *excess_source, *excess_sink);
                //printf("%d\n",cycle);
                
                //printf("Node: %d, cycle: %d\n", node_i, cycle);
            }
            cycle = cycle - 1;
        }
    }
}

// GPU Kernels: Multi-band Blending (GPU-accelerated)
__global__ void blendLaplacianGPU_kernel(
    const short* lapA,      
    const short* lapB,      
    const float* mask,      
    short* output,        
    int width, int height, int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    float m = mask[idx];  // Mask value [0,1]
    float inv_m = 1.0f - m;
    
    // Blend từng channel
    for (int c = 0; c < channels; c++) {
        int pixel_idx = idx * channels + c;
        float a = (float)lapA[pixel_idx];
        float b = (float)lapB[pixel_idx];
        output[pixel_idx] = (short)(a * inv_m + b * m);
    }
}

// Kernel downsample mask (Gaussian pyramid) - thay pyrDown
__global__ void downsampleMaskGPU_kernel(
    const float* src, float* dst,
    int src_width, int src_height,
    int dst_width, int dst_height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    // Average 2x2 block với boundary check
    int sx = x * 2;
    int sy = y * 2;
    
    float sum = 0.0f;
    int count = 0;
    
    for (int dy = 0; dy < 2; dy++) {
        for (int dx = 0; dx < 2; dx++) {
            int nx = sx + dx;
            int ny = sy + dy;
            if (nx < src_width && ny < src_height) {
                sum += src[ny * src_width + nx];
                count++;
            }
        }
    }
    
    dst[y * dst_width + x] = (count > 0) ? (sum / count) : 0.0f;
}

// Kernel add 2 ảnh (cho reconstruct Laplacian)

__global__ void addLaplacianGPU_kernel(
    const float* upsampled,  
    const short* laplacian, 
    float* output,           
    int width, int height, int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    for (int c = 0; c < channels; c++) {
        int pixel_idx = idx * channels + c;
        output[pixel_idx] = upsampled[pixel_idx] + (float)laplacian[pixel_idx];
    }
}

// Kernel upsample 2x với bilinear interpolation
__global__ void upsampleGPU_kernel(
    const float* src, float* dst,
    int src_width, int src_height,
    int dst_width, int dst_height, int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= dst_width || y >= dst_height) return;
    
    // Bilinear interpolation
    float fx = (float)x / 2.0f;
    float fy = (float)y / 2.0f;
    
    int x0 = (int)fx;
    int y0 = (int)fy;
    int x1 = min(x0 + 1, src_width - 1);
    int y1 = min(y0 + 1, src_height - 1);
    
    float dx = fx - x0;
    float dy = fy - y0;
    
    int dst_idx = y * dst_width + x;
    
    for (int c = 0; c < channels; c++) {
        float v00 = src[(y0 * src_width + x0) * channels + c];
        float v01 = src[(y0 * src_width + x1) * channels + c];
        float v10 = src[(y1 * src_width + x0) * channels + c];
        float v11 = src[(y1 * src_width + x1) * channels + c];
        
        float v0 = v00 * (1.0f - dx) + v01 * dx;
        float v1 = v10 * (1.0f - dx) + v11 * dx;
        
        dst[dst_idx * channels + c] = v0 * (1.0f - dy) + v1 * dy;
    }
}
