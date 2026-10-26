# Kết quả kiểm tra triển khai — 25–26/10/2026

## Huấn luyện trên Kaggle — 26/10/2026

Một lần chạy [Kaggle_Train.ipynb](../model_training/Kaggle_Train.ipynb) trên Tesla T4, Python 3.13.15, Torch 2.11.0+cu128, Ultralytics 8.3.170; imgsz 640, batch 16, tối đa 100 epoch, patience 15, seed 42. Số liệu gốc, đường cong và confusion matrix nằm trong [training/](training/).

| Góc nhìn | Dataset | Lớp | Ảnh train / val / test |
|---|---|---|---|
| Dưới nước | [Figshare Underwater Drowning Detection](https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235), CC BY 4.0 | swimming, struggling, drowning | 4.488 / 1.125 / — |
| Trên mặt nước | [Roboflow Swimming and Drowning Detection](https://universe.roboflow.com/university-g3h71/swimming-and-drowning-detection), CC BY 4.0 | drowning, out_of_water, swimming | 10.140 / 1.471 / 744 (đã loại 10 ảnh trùng chính xác khỏi val/test) |

### Dưới nước (chỉ có validation)

| Mô hình | Epoch chạy (tốt nhất) | mAP50 | mAP50-95 | swimming R | struggling P / R | drowning P / R |
|---|---|---|---|---|---|---|
| YOLO11n | 19 (4) | 0,755 | 0,469 | 0,392 | 0,545 / 0,760 | 0,685 / 1,000 |
| **YOLO11s — triển khai** | 24 (9) | **0,816** | **0,570** | 0,465 | 0,882 / 0,872 | 0,765 / 0,949 |

- Loss validation tăng ngay từ epoch 1 trong khi loss train giảm đều; mAP validation dao động 0,57–0,82 giữa các epoch. Tập val gồm các đoạn video liền mạch (ví dụ frame_67976…68043 cùng một người) khác cảnh với train, nên đây là đánh giá khó và nhiễu; checkpoint tốt nhất là một đỉnh của đường dao động, nên con số còn lạc quan.
- Confusion matrix (YOLO11s): 31% box swimming thật bị dự đoán drowning, 19% bị bỏ sót; struggling 92% đúng, drowning 95% đúng. Ảnh val nhãn drowning có tư thế úp mặt, duỗi tay giống lướt nước bình thường, nên dữ liệu vốn khó phân biệt.
- Kết quả thấp hơn rõ so với 97,5% mAP50 tác giả dataset công bố cho YOLOv8n (không rõ cách chia của họ); không dùng con số 97,5% để so sánh.
- Quy tắc chọn ban đầu (recall drowning trong 1 điểm của tốt nhất) chọn YOLO11n vì recall 1,000, nhưng YOLO11n đạt được nhờ dự đoán drowning tràn lan (precision 0,685, struggling precision 0,545). Quy tắc được sửa sau lần chạy này: chỉ xét mô hình có mAP50 trong 2 điểm của tốt nhất, rồi mới xét recall/precision drowning và độ trễ. Theo quy tắc mới chọn YOLO11s. Việc sửa dựa trên kết quả validation, không có tập test nào bị dùng để chọn; `training/underwater/selected.json` vẫn ghi lựa chọn cũ của notebook.

### Trên mặt nước

| Mô hình | Epoch chạy (tốt nhất) | val mAP50 | val mAP50-95 | drowning P / R | out_of_water P / R | swimming P / R |
|---|---|---|---|---|---|---|
| YOLO11n | 100 (69) | 0,889 | 0,567 | 0,854 / 0,878 | 0,832 / 0,810 | 0,805 / 0,867 |
| **YOLO11s — triển khai** | 100 (96) | **0,898** | **0,599** | 0,879 / 0,882 | 0,883 / 0,844 | 0,827 / 0,864 |
| **YOLO11s trên test** | — | **0,873** | **0,562** | 0,859 / 0,878 | 0,843 / 0,756 | 0,867 / 0,850 |

- Test được chạy một lần sau khi chọn mô hình. Split của Roboflow chia ngẫu nhiên theo ảnh; audit tìm thấy 164.042 cặp ảnh gần trùng giữa các tập, nên test vẫn có thể chứa frame cùng cảnh với train và không thay cho đánh giá trên video độc lập.
- out_of_water chỉ có 98 box val / 71 box test; số liệu lớp này kém tin cậy.

### Chạy thử trên video demo (CPU)

Hai checkpoint được chép vào `fastapi-backend/model/detection_model/` (`underwater_selected.pt` = YOLO11s dưới nước, `abovewater_selected.pt` = YOLO11s trên mặt nước); backend tự dùng chúng. Chạy `tools/benchmark.py` trên toàn bộ 7 video demo, GAN tắt: tất cả hoàn thành, khoảng 8,5–9,4 FPS, YOLO khoảng 80–84 ms/frame trên CPU.

Các video demo trong repo là **video đầu ra của hệ thống gốc**: khung, nhãn ("normal 0.90", "ID:23 drowning") và số "Active Tracks" đã in cứng vào hình, và không có nhãn sự kiện. Vì vậy chúng chỉ dùng để chạy thử, không dùng để đánh giá. Ghi nhận định tính: mô hình trên mặt nước mới bật `drowning_alert` ở `ZoneB_Abovewater` (6,6 s) và `ZoneA_Abovewater(1)` (7,5 s), nơi người trong hình có vẻ đang bơi sải bắn nước, nhiều khả năng là báo động giả; mô hình dưới nước mới phát hiện người thưa hơn mô hình cũ trên các video này (khác camera/miền dữ liệu so với Figshare). Cần video gốc có gán nhãn để kết luận.

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

- Chưa có metadata nhóm nguồn cho cả hai dataset để xác nhận tính độc lập giữa các tập.
- Chưa có video gốc (không có overlay) gán nhãn sự kiện để hiệu chỉnh ngưỡng hoặc công bố cảnh báo sai/phút, bỏ sót và độ trễ sự kiện; các video demo hiện có không dùng được cho mục đích này.
- Chưa chạy FUnIE-GAN thật hoặc so sánh GAN bật/tắt; các nhánh tích hợp GAN chỉ được kiểm thử bằng giả lập.
- Chưa chạy nguồn liên tục đủ 30 phút, camera vật lý hay kiểm tra âm thanh trên thiết bị thật. Công cụ soak và hướng dẫn đã được chuẩn bị.

Các bước còn lại, câu lệnh và định dạng dữ liệu nằm trong [IMPLEMENTATION.md](IMPLEMENTATION.md). Hai checkpoint YOLO gốc được giữ nguyên để rollback; không có số liệu huấn luyện hoặc nghiệm thu nào được tạo giả.
