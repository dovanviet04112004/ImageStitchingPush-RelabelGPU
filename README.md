# push_relabel_gpu

Mục tiêu
- Ghép ảnh theo chiều ngang dùng GPU: dò overlap → graph-cut (push-relabel) → multi-band blending.

Yêu cầu môi trường (phiên bản tham khảo)
- Hệ điều hành: Windows 10/11
- CUDA Toolkit (nvcc) tương thích C++14 (tested với nvcc đi kèm CUDA 10/11)
- OpenCV 4.1.2 (sử dụng lib `opencv_world4120`) 

Các file 
- `gpu-project.cpp`
- `cudacut.cu`
- `CudaCut_kernel.cu`
- `cudacut.cuh`
- `README.md` 

Hướng dẫn build (từ thư mục gốc `d:\push_relabel_gpu`)
```powershell
nvcc -std=c++14 -O2 -w gpu-project.cpp cudacut.cu -o gpu_project.exe -IC:\opencv\build\include -LC:\opencv\build\x64\vc16\lib -lopencv_world4120
```
Nếu OpenCV cài ở nơi khác hoặc phiên bản khác, thay `-I`/`-L` và tên thư viện cho phù hợp.

Chạy (ví dụ)
- Chạy mặc định (2 ảnh test):
```powershell
.\gpu_project.exe
```
- Chạy preset `2k` (nếu có file mẫu):
```powershell
.\gpu_project.exe 2k
```

Mô tả các module
- `gpu-project.cpp`: Entry chính của chương trình. Xử lý CLI/preset, tải ảnh, chuyển sang grayscale cho phần tính toán seam, gọi `autoDetectOverlapGPU` để xác định độ chồng lắp, gọi hàm ghép từng cặp ảnh (khởi tạo graph, `push_relabel`, `selectPix`) và cuối cùng gọi `multiBandBlendGPU`. In các thống kê thời gian và `GPU peak used`, hiển thị và lưu ảnh kết quả.
- `cudacut.cu`: Triển khai lớp `CudaCut` — quản lý bộ nhớ host/device (cudaMalloc / cudaMallocManaged / free), khởi tạo trọng số đồ thị (weights), thiết lập dữ liệu cho kernel và điều phối việc gọi các kernel CUDA; chứa các biến thể xử lý (ví dụ `global_relabel_CPU`) và các hàm bọc gọi tới kernel.
- `CudaCut_kernel.cu`: Tập hợp các kernel CUDA thực sự: `setupGraph_kernel`, `adjustGraph_kernel`, `push_relabel_kernel`, `computeOverlapDiff_kernel`, `selectPixGPU_kernel`, các kernel blending/seam, và helper kernel khác.
- `cudacut.cuh`: Header chứa khai báo lớp `CudaCut`, prototype các hàm và kernel, macro đo thời gian, và helper kiểm tra lỗi CUDA.