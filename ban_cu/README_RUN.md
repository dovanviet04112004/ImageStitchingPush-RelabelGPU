# Chạy bản legacy (ban_cu)

Tập tin này hướng dẫn nhanh cách build và chạy phiên bản legacy trong thư mục `ban_cu`.

Yêu cầu môi trường
- CUDA toolkit (nvcc) trong PATH
- OpenCV (binaries) - ví dụ đã dùng: `C:\opencv\build`
- Windows PowerShell (hoặc CMD)

Ảnh đầu vào
- Đặt 2 ảnh chồng vào `images/APAP dataset/` (ví dụ `image_1_1008x755.png`, `image_2_1008x755.png`).
- Code gốc dùng macro trong `cudacut.cuh`: `WIDTH=1008`, `HEIGHT=755`, `OVERLAP_WIDTH=100`.
- Hai ảnh nên cùng chiều cao (755) và có ít nhất `OVERLAP_WIDTH` cột.

Build (từ thư mục gốc `d:\push_relabel_gpu`)
```powershell
nvcc -std=c++14 -O2 -w ban_cu\gpu-project.cpp ban_cu\cudacut.cu -o ban_cu_gpu.exe -IC:\opencv\build\include -LC:\opencv\build\x64\vc16\lib -lopencv_world4120
```

Chạy (từ thư mục gốc)
```powershell
.\ban_cu_gpu.exe
# hoặc chạy trực tiếp trong thư mục ban_cu
# .\ban_cu\ban_cu_gpu.exe
```

Gợi ý và lỗi thường gặp
- "can't open/read file": kiểm tra đường dẫn và tên file trong `gpu-project.cpp`.
- Nếu chương trình chạy nhưng `max_flow: 0`: kiểm tra overlap (các pixel giống nhau → không có seam).
- Để dùng ảnh kích thước khác, sửa macro trong `ban_cu/cudacut.cuh` hoặc sửa code để dùng `A.cols/A.rows` runtime.

Tệp kết quả
- Các cửa sổ hiển thị `result` và `result1_stitching` (OpenCV imshow). Bạn có thể lưu thủ công trong `gpu-project.cpp`.