#include "cudacut.cuh"


// push kernel
inline __global__ void
push_relabel_kernel(int* d_right_weight, int* d_left_weight, int* d_up_weight, int* d_down_weight,
    int* d_excess_flow, int* d_graph_height, int* d_relabel_mask, int* d_height_backup,
    int width, int height, int N, int* excess_source, int* excess_sink) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int node_i = iy * width + ix;

    if (node_i < N && ix != 0 && (ix + 1) != width) {
        int cycle = 1024;
        int h_dash, v_dash, d;

        while (cycle > 0) {
            if (d_excess_flow[node_i] > 0 && d_graph_height[node_i] < N) {
                h_dash = 1 << 22;
                v_dash = -1;

                if (iy > 0 && iy < height - 1) {
                    int v_list[4] = { node_i - 1, node_i + 1, node_i - width, node_i + width };
                    for (int i = 0; i < 4; i++) {
                        int v = v_list[i];
                        
                        if (v >= 0 && v < N) {
                            //printf("node_i_push: %d, v: %d, h_dash: %d, d_graph_height[v]: %d, d_graph_height[node_i]: %d\n", node_i, v, h_dash, d_graph_height[v], d_graph_height[node_i]);
                            if (d_graph_height[v] < h_dash ) {
                                //printf("2\n");
                                h_dash = d_graph_height[v];
                                v_dash = v;
                            }
                        }
                    }
                    //printf("node_i: %d, v_dash: %d, excess_flow: %d\n", node_i, v_dash, d_excess_flow[node_i]);
                }
                else if (iy == 0) {
                    int v_list[3] = { node_i - 1, node_i + 1, node_i + width };
                    for (int i = 0; i < 3; i++) {
                        int v = v_list[i];
                        if (v >= 0 && v < N) {
                            if (d_graph_height[v] < h_dash ) {
                                //printf("2\n");
                                h_dash = d_graph_height[v];
                                v_dash = v;
                            }
                        }
                    }
                    //printf("node_i: %d, v_dash: %d, excess_flow: %d\n", node_i, v_dash, d_excess_flow[node_i]);
                }
                else if (iy == height - 1) {
                    int v_list[3] = { node_i - 1, node_i + 1, node_i - width };
                    for (int i = 0; i < 3; i++) {
                        int v = v_list[i];
                        if (v >= 0 && v < N) {
                            if (d_graph_height[v] < h_dash ) {
                                //printf("2\n");
                                h_dash = d_graph_height[v];
                                v_dash = v;
                            }
                        }
                    }
                    //printf("node_i: %d, v_dash: %d, excess_flow: %d\n", node_i, v_dash, d_excess_flow[node_i]);
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

