#include "cudacut.cuh"
#include "CudaCut_kernel.cu"

CudaCut::CudaCut(int image_width, int image_height, int overlap_width)
    : height(image_height), width(overlap_width), image_width(image_width) {
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

__global__ void
setupGraph_kernel(int* d_horizontal, int* d_vertical, int* d_right_weight, int* d_left_weight, int* d_up_weight,
    int* d_down_weight, int* d_excess_flow, int* d_push_block_position, int* d_graph_height, int* d_relabel_mask,
    int width, int height, int N) {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    int node_i = iy * width + ix;
    
    
    d_right_weight[node_i] = d_horizontal[iy * (width ) + ix + 1];
    d_left_weight[node_i] = d_horizontal[iy * (width ) + ix];

    d_down_weight[node_i] = d_vertical[(iy + 1) * width + ix];
    d_up_weight[node_i] = d_vertical[node_i];
    d_excess_flow[node_i] = 0;
    d_graph_height[node_i] = width - ix - 1;
    d_relabel_mask[node_i] = 0;
    /*if ((node_i + 2) % width == 0) {
        printf(" node_i: %d, d_right_weight: %d, d_left_weight: %d\n", node_i, d_right_weight[node_i], d_left_weight[node_i]);
    }*/
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
        //printf("d_excess_flow[%d]: %d,d_right: %d\n", node_i, d_excess_flow[node_i], d_right_weight[node_i]);
    }
}


int CudaCut::cudaCutsSetupGraph(cv::Mat& image1, cv::Mat& image2) {
    int xoffset = WIDTH - OVERLAP_WIDTH;
    image1(cv::Rect(xoffset, 0, OVERLAP_WIDTH, HEIGHT)).copyTo(area1);
    image2(cv::Rect(0, 0, OVERLAP_WIDTH, HEIGHT)).copyTo(area2);

    // Initialize horizontal weights
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width - 1; x++) {
            uchar a0 = image1.at<uchar>(y, xoffset + x);
            uchar b0 = image2.at<uchar>(y, x);
            uchar cap0 = abs(a0 - b0);

            uchar a1 = image1.at<uchar>(y, xoffset + x + 1);
            uchar b1 = image2.at<uchar>(y, x + 1);
            uchar cap1 = abs(a1 - b1);
            //cout << "y = " << y << " x = " << x << "" << endl;
            //cout << y * (width)+x + 1 << endl;
            d_horizontal[y * (width) + x + 1] = static_cast<int>(cap0 + cap1);
        }
        //cout << y * (width) + width << endl;
        //cout << y * (width) << endl;
        d_horizontal[y * (width) + width] = 0;
        d_horizontal[y * (width)] = 0;
        //cout << endl;
    }


    // Initialize vertical weights
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height - 1; y++)
        {
            uchar a0 = image1.at<uchar>(y, xoffset + x);
            uchar b0 = image2.at<uchar>(y, x);
            uchar cap0 = abs(a0 - b0);

            uchar a1 = image1.at<uchar>(y + 1, xoffset + x);
            uchar b1 = image2.at<uchar>(y + 1, x);
            uchar cap1 = abs(a1 - b1);
            d_vertical[(y + 1) * width + x] = (int)(cap0 + cap1);
            //            vertical[y * graph.width + x] = 0;
        }
        d_vertical[(height) * width + x] = 0;
        d_vertical[x] = 0;
    }
    dim3 block(16, 8, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    //cout << "<<<grid " << grid.x << grid.y << " block " << block.x << block.y << ">>>" << endl;
    setupGraph_kernel << <grid, block >> > (d_horizontal, d_vertical, d_right_weight, d_left_weight, d_up_weight,
        d_down_weight, d_excess_flow, d_push_block_position, d_graph_height, d_relabel_mask, width, height, graph_size);
    adjustGraph_kernel << <grid, block >> > (d_excess_flow, d_left_weight, d_right_weight, d_down_weight, d_up_weight, d_graph_height,
        d_up_right_sum, d_up_left_sum, d_down_right_sum, d_down_left_sum, width, height, graph_size, Excess_total);
    
    cudaDeviceSynchronize();
    cout<<"setup graph done"<<endl;
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

void CudaCut::global_relabel() {
    /*for (int node = 0; node < graph_size; node++) {
        if (d_excess_flow[node] > 0) {
            printf("node: %d, d_excess_flow: %d\n", node, d_excess_flow[node]);
        }
    }*/
    for (int node = 0; node < graph_size; node++) {
        /*if (d_excess_flow[node] > 0) {
            printf("node: %d, d_excess_flow before: %d\n", node, d_excess_flow[node]);
        }*/
        int x = node % width;
        
        int y = node / width;
        //printf("node: %d, excess_flow: %d, d_right: %d\n", node, d_excess_flow[node], d_right_weight[node]);
        
        //if (d_excess_flow[node] > 0) {
        //    printf("node: %d, d_excess_flow after: %d\n", node, d_excess_flow[node]);
        //}
        
        
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
                if(nei == node + 1 && d_graph_height[node] > d_graph_height[node + 1] && d_right_weight[node] > 0) {
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
    
    //cout << "global-relabel-1 done\n";


    int current;

    for(int i = 0; i < graph_size; i++) {
        scanned[i] = false;
	}

    queue<int> que;
    int node1 = width - 1;
    int s;
    //int count = 0;
    for (int i = 0; i < height; i++) {
        que.push(node1 + i * width);
        scanned[node1 + i * width] = true;
        d_graph_height[node1 + i * width] = 0;
    }

    while (!que.empty()) {
        s = que.front();

        que.pop();

        current = d_graph_height[s];

        current = current + 1;
        int x, y;
        x = s % width;
        y = s / width;
        //cout << "h_right_weight[s - 1]:" << h_right_weight[s - 1] << endl;
        //cout << "h_up_weight[s + width]:" << h_up_weight[s + width] << endl;
        //cout << "h_down_weight[s - width]:" << h_down_weight[s - width] << endl;
        //cout << "h_left_weight[s + 1]:" << h_left_weight[s + 1] << endl;
        //cout << "s: " << s << ", s - 1: " << s - 1 << ", d_right_weight[s - 1]: " << d_right_weight[s - 1] << ", scanned[s - 1]: " << scanned[s - 1] << ", x: "<<x<<", y: "<<y << endl;
        if (x != 0 && d_right_weight[s - 1] > 0 && scanned[s - 1] == false) {
            //cout << "s_1: " << s << ", s - 1: " << s - 1 << ", d_right_weight[s - 1]: " << d_right_weight[s - 1] << ", scanned[s - 1]: " << scanned[s - 1] << endl;
            d_graph_height[s - 1] = current;
            scanned[s - 1] = true;
            que.push(s - 1);
            //count++;
        }
        if (y < height - 1 && d_up_weight[s + width] > 0 && scanned[s + width] == false) {
            //cout << "s: " << s << ", s + width: " << s + width << ", d_up_weight[s + width]: " << d_up_weight[s + width] << ", scanned[s + width]: " << scanned[s + width] << endl;
            d_graph_height[s + width] = current;
            scanned[s + width] = true;
            que.push(s + width);
            //count++;
        }
        if (y >= 1 && d_down_weight[s - width] > 0 && scanned[s - width] == false) {
            //cout << "s: " << s << ", s - width: " << s - width << ", d_down_weight[s - width]: " << d_down_weight[s - width] << ", scanned[s - width]: " << scanned[s - width] << endl;
            d_graph_height[s - width] = current;
            scanned[s - width] = true;
            que.push(s - width);
            //count++;
        }
        if (x < width - 1 && d_left_weight[s + 1] > 0 && scanned[s + 1] == false) {
            //cout << "s: " << s << ", s + 1: " << s + 1 << ", d_left_weight[s + 1]: " << d_left_weight[s + 1] << ", scanned[s + 1]: " << scanned[s + 1] << endl;
            d_graph_height[s + 1] = current;
            scanned[s + 1] = true;
            que.push(s + 1);
            //count++;
        }
    }

    //cout<< "global-relabel-2 done\n";

    bool if_all_are_relabeled = true;
    //int check = 0;
    for (int i = 0; i < graph_size; i++)
    {
        if (scanned[i] == false )
        {
            //check++;
            //printf("i_false: %d\n", i);
            if_all_are_relabeled = false;
        }
    }
    /*for (int i = 0; i < graph_size; i++)
    {
        if (scanned[i] == true)
        {
            check++;
            printf("i_true: %d\n", i);
        }
    }
    cout << "check: " << check << endl;*/

    // if not all nodes are relabeled
    if (if_all_are_relabeled == false)
    {
        // for all nodes
        for (int i = 0; i < graph_size; i++)
        {
            // if i'th node is not marked or relabeled
            if (scanned[i] == false && i % width != 0 && (i + 1) % width != 0)
            {
                // mark i'th node
                //mark[i] = true;

                /* decrement excess flow of i'th node from Excess_total
                 * This shows that i'th node is not scanned now and needs to be marked, thereby no more contributing to Excess_total
                 */

                *Excess_total = *Excess_total - d_excess_flow[i];
                d_excess_flow[i] = 0;
            }
        }
    }



}

void CudaCut::push_relabel()
{
    for (int i = 0; i < graph_size; i++)
    {
        mark[i] = false;
    }
    dim3 block(32, 8, 1);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y, 1);
    int count = 0;
    int k = 3;
    while ((*excess_source + *excess_sink) < (*Excess_total))
    {
        int cycle = 1024;
            push_relabel_kernel << <grid, block >> > (d_right_weight, d_left_weight, d_up_weight, d_down_weight,
                d_excess_flow, d_graph_height, d_relabel_mask, d_height_backup,
                width, height, graph_size, excess_source, excess_sink);

            cudaDeviceSynchronize();
        *excess_source = 0;
        *excess_sink = 0;
        global_relabel();
    }

    cout << "max_flow: " << *excess_sink << endl;
}

