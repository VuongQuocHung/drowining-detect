# Kết quả kiểm tra triển khai — 25/10/2026

## Đã thực hiện

- Backend: 17 kiểm thử về thời gian cảnh báo, mất dấu, khôi phục, tách người/phiên, cập nhật nhãn DeepSORT, frame trống, upload, 409, stop/restart, SSE/MJPEG reconnect, download nhiều lần, ảnh tĩnh, camera mất kết nối giả lập và thiếu GAN. Enhancement video dùng chung worker được kiểm tra với GAN giả lập.
- Pipeline dữ liệu/đánh giá: 5 kiểm thử với dataset tổng hợp, gồm nhãn lỗi, trùng ảnh, nhóm nguồn, chọn mô hình và chấm sự kiện.
- Frontend: 4 kiểm thử với SSE và AudioContext giả lập, gồm khôi phục phiên, âm cảnh báo theo thao tác người dùng, chống lặp pulse, lỗi 409 và ảnh tĩnh.
- `npm run lint`, `npm run build` có kiểm tra TypeScript; compile Python và kiểm tra cú pháp các cell notebook.
- Nạp hai checkpoint YOLO cũ và trọng số ReID DeepSORT thật; kiểm tra theo dõi trên ảnh mẫu. Xác nhận checkpoint dưới nước cũ có lớp `0=drowning`, `1=normal`, không cùng thứ tự với dataset mới.
- Chạy worker thật với 12 frame đầu của `ZoneA_Underwater.mp4`, GAN tắt: hoàn thành, không có lỗi. Báo cáo máy đọc được ở [benchmark-smoke.json](benchmark-smoke.json).

## Benchmark ngắn

| Thuộc tính | Giá trị |
|---|---|
| Hệ điều hành | Windows, build 26200 |
| Python | 3.11.9 |
| PyTorch thực tế | 2.14.0+cpu |
| Ultralytics | 8.3.170 |
| OpenCV thực tế | 5.0.0 |
| Mô hình | underwaterModel.pt cũ, confidence 0,5 |
| Nguồn | 12 frame đầu video demo ZoneA_Underwater |
| GAN | Tắt |
| FPS đo được | 6,00 |
| Độ trễ p50 / p95 | 140,55 / 282,42 ms |
| RAM tiến trình tại mẫu cuối | 659,73 MB |

Đây là smoke benchmark ngắn trên CPU, không phải cam kết FPS, phép đo ổn định dài hạn hay đánh giá chất lượng nhận diện. Các số CPU từ psutil có thể vượt 100% vì tính trên nhiều lõi. Môi trường Python có sẵn khác các bản Torch/OpenCV đã ghim cho quy trình tái lập; chưa xác minh bằng cách cài lại toàn bộ môi trường ghim.

## Chưa thực hiện

- Chưa tải/kiểm tra toàn bộ dataset Figshare trong workspace và chưa có metadata nhóm nguồn để xác nhận tính độc lập giữa các tập.
- Chưa chạy huấn luyện trên tài khoản GPU Kaggle; vì vậy chưa có checkpoint mới được chọn, mAP thực nghiệm mới hay báo cáo so sánh YOLO11n/YOLO11s.
- Chưa có video độc lập gán nhãn sự kiện để hiệu chỉnh ngưỡng hoặc công bố cảnh báo sai/phút, bỏ sót và độ trễ sự kiện.
- Chưa chạy FUnIE-GAN thật hoặc so sánh GAN bật/tắt; các nhánh tích hợp GAN chỉ được kiểm thử bằng giả lập.
- Chưa chạy nguồn liên tục đủ 30 phút, camera vật lý hay kiểm tra âm thanh trên thiết bị thật. Công cụ soak và hướng dẫn đã được chuẩn bị.

Các bước còn lại, câu lệnh và định dạng dữ liệu nằm trong [IMPLEMENTATION.md](IMPLEMENTATION.md). Hai checkpoint YOLO gốc được giữ nguyên để rollback; không có số liệu huấn luyện hoặc nghiệm thu nào được tạo giả.
