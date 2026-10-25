# Chạy và đánh giá bản tối ưu

Ứng dụng xử lý một nguồn mỗi lần. Surveillance và Drowning Detection dùng cùng giao diện và cùng phiên backend; đổi tab không khởi chạy nguồn thứ hai. Webcam là camera trên máy chạy backend.

## Cài đặt

Dùng Python 3.10 hoặc 3.11. Từ `fastapi-backend/`:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tools/fetch_tracker.py
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --workers 1
```

`fetch_tracker.py` tải checkpoint từ thư mục Google Drive được [README DeepSORT chính thức](https://github.com/ZQPei/deep_sort_pytorch#quick-start) liên kết và kiểm tra SHA256. Không dùng trọng số khởi tạo ngẫu nhiên để thay thế ReID. Chạy một worker Uvicorn vì registry và khóa suy luận nằm trong bộ nhớ của tiến trình.

Từ `frontend/`, chạy `npm ci`, `npm run dev`. Có thể đặt `VITE_API_URL` theo `.env.example`. Tài khoản demo hiện có: `lifeguard@admin.com` / `password123`; cơ chế đăng nhập vẫn là mock của đồ án.

GAN mặc định tắt và không được nạp lúc khởi động. Muốn thử GAN, cài `requirements-gan.txt` trong môi trường riêng rồi bật checkbox. Khả năng chạy GPU của TensorFlow phụ thuộc hệ điều hành/môi trường; đo thực tế trước khi bật mặc định. Thiếu GAN không ngăn pipeline YOLO chạy. Môi trường optional GAN này chưa được kiểm chứng bằng model thật trong phiên triển khai.

## Hành vi nhận diện

- Tên lớp từ checkpoint được chuẩn hóa: `normal`, `swimmer` → `swimming`; giữ `struggling`, `drowning`, `float`; nhãn khác → `unknown`. `float` (vật nổi) chỉ được vẽ lên khung hình, không theo dõi, không cảnh báo, không có trong danh sách người. `class_id` trả về là ID nội bộ: swimming=0, struggling=1, drowning=2, unknown=3, không phải ID gốc checkpoint.
- DeepSORT riêng cho từng phiên, áp dụng cả video dưới nước và trên mặt nước. Track phải được xác nhận trước khi hiển thị. Nhãn và confidence cập nhật theo detection hiện tại; frame trống vẫn cập nhật tuổi track.
- Cảnh báo từng người: suspected → struggling_alert sau 2 giây struggling, drowning_alert sau 1 giây drowning. Khoảng mất dấu ≤0,5 giây giữ bằng chứng nhưng không cộng thời gian; dài hơn thì xóa thời gian đang xác nhận. Nhận diện swimming phải liên tục 2 giây để hạ cảnh báo đã bật.
- Mất dấu luôn hiện `lost`; `previous_level` giữ mức trước đó, không tự coi người đã an toàn. Track mất dấu được giữ tối đa 30 giây để giới hạn bộ nhớ.
- Video dùng timestamp của video, fallback frame index/FPS nếu timestamp không tăng. Camera dùng monotonic clock và hàng đợi một frame mới nhất. Video upload đọc tuần tự, nghỉ theo thời gian xử lý còn lại của chu kỳ FPS.
- Âm thanh trình duyệt cần bấm “Bật âm cảnh báo”; chỉ phát khi backend gửi `alarm=true`. Nhắc lại sau 10 giây. SSE khôi phục lần đầu không phát lại còi cũ. Có thể export `BACKEND_ALARM=1` để phát thêm trên máy server.
- Ảnh tĩnh chỉ phân loại, không tạo bộ đếm hoặc cảnh báo theo thời gian. Confidence không được diễn giải thành xác suất người thực sự đuối nước.

Các mặc định là tham số thử nghiệm, chưa được hiệu chỉnh trên video gán nhãn. Thay đổi bằng biến môi trường trong `fastapi-backend/.env.example`; backend không tự đọc file `.env`. Dừng và tạo phiên mới sau khi đổi cấu hình/mô hình.

## API và vòng đời

Giữ các endpoint upload, webcam, MJPEG, download và enhancement. `/webcam/start` bổ sung `session_id`, `events_url`; upload video trả session ngay và worker chạy độc lập với người xem. Lỗi mở camera/video/GAN chuyển phiên sang `failed` và được gửi qua SSE/status. Nguồn thứ hai nhận HTTP 409. Đổi confidence khi nguồn đang hoạt động cũng nhận 409; truyền ngưỡng mới khi tạo phiên.

- `GET /sessions/active`: phiên đang hoạt động hoặc null.
- `GET /video_session_status/{id}`: trạng thái, lỗi, số frame và khả năng download.
- `GET /sessions/{id}/events`: SSE gồm `sequence`, `session_id`, `source_time`, `detections`, `metrics`, `status`. Detection có `track_id`, `class`, `confidence`, `alert_level`, `alarm`.
- `Last-Event-ID` hoặc `?after=` khôi phục tối đa 300 sự kiện còn trong buffer. Không có cursor thì nhận trạng thái mới nhất và bỏ pulse còi cũ. Mất kết nối lâu hơn buffer không đảm bảo replay mọi sự kiện; JSONL giữ lịch sử đầy đủ.
- `GET /realtime_video_stream/{id}` và `/video_feed`: xem lại JPEG mới nhất, không chạy suy luận lại.
- `POST /stop_video_processing/{id}`: dừng cả phiên webcam/video/enhancement. Đọc status để xác nhận kết thúc nếu driver camera chậm phản hồi.

Worker giải phóng capture/writer trong finally, xóa upload sau kết thúc. Video hoàn tất tải được nhiều lần; kết quả, JSONL và registry hết hạn sau SESSION_TTL (mặc định 1 giờ). Dừng giữa chừng xóa video output chưa hoàn chỉnh, giữ JSONL để chẩn đoán. SSE và MJPEG không sở hữu vòng đời worker.

Enhancement video cũng dùng chung worker và giới hạn một nguồn; endpoint comparison/enhanced-only chỉ đọc kết quả worker. Các file ảnh enhancement cũng hết hạn theo TTL. Ứng dụng không lưu registry qua lần restart; các artifact từ tiến trình bị kết thúc đột ngột cần dọn thủ công sau khi server đã dừng.

## Kaggle: dữ liệu và huấn luyện

Mở [Kaggle_Train.ipynb](../model_training/Kaggle_Train.ipynb) trên Kaggle, chọn Accelerator GPU (T4 x2 hoặc P100), bật Internet, thêm Secret `ROBOFLOW_API_KEY` (Add-ons → Secrets) rồi Run All. Notebook train lần lượt hai góc nhìn trong `VIEWS`; xóa một phần tử để chỉ train góc còn lại. Notebook tự sparse-clone thư mục `model_training/pipeline` từ GitHub và tải dataset vào `/tmp`, nên không cần gắn input và dataset không bị lưu vào output. Key Roboflow chỉ đọc từ Kaggle Secrets, không nằm trong code hay output.

| Góc nhìn | Dataset | Lớp |
|---|---|---|
| `underwater` | [Figshare Underwater Drowning Detection Dataset](https://figshare.com/articles/dataset/Underwater_Drowning_Detection_Dataset/29497235?file=56039351), CC BY 4.0 | swimming / struggling / drowning |
| `abovewater` | [Roboflow water-rescue-dataset v1](https://universe.roboflow.com/cuizh/water-rescue-dataset/dataset/1), CC BY 4.0, tải bản export YOLOv11 qua API | Drowning / float / swimmer → drowning / float / swimming |

Notebook dùng `requirements-kaggle.txt` để giữ Torch/TorchVision/CUDA có sẵn của Kaggle. Không dùng `requirements.txt` cục bộ trong Kaggle Python 3.13: TorchVision 0.20.1 và NumPy 1.26.4 không hỗ trợ Python này. Chạy cell kiểm tra GPU/NMS trước audit; phiên bản thực tế được lưu trong environment.json và installed-packages.txt. Muốn resume cần giữ môi trường tương thích với lần chạy trước. Requirements đầy đủ ở backend vẫn dành cho môi trường local Python 3.10/3.11.

Dataset dưới nước có 5.613 ảnh, ba lớp swimming/struggling/drowning, chia sẵn train 4.488 / val 1.125, không có test. Đã kiểm tra trước toàn bộ nhãn (25/10/2026): mỗi ảnh một box, 1.871 box mỗi lớp (train 1.496, val 375), không có nhãn lỗi; trùng ảnh giữa tập vẫn do audit xác minh trên Kaggle. Audit kiểm tra ảnh/nhãn YOLO, class ID, tọa độ, box vượt ảnh; tạo preview, class counts, SHA256 pixel và dHash. Tên lớp đọc từ `classes.txt` hoặc `data.yaml` và chuẩn hóa (`normal`, `swimmer` → `swimming`); `float` là vật nổi (phao, ván…), được train như một lớp riêng để mô hình không nhầm thành người; tên khác phải ánh xạ rõ bằng `--rename OLD=NEW` (trong notebook: `DATA[view]['rename']`). Audit nhận cả cấu trúc `images/{train,val}` (Figshare) và `{train,valid,test}/images` (Roboflow). Nhãn lỗi chặn chuẩn bị dữ liệu; ảnh trùng chính xác giữa tập cũng chặn, trừ khi bật `--drop-exact-duplicates` (notebook bật mặc định): bản ở val/test bị loại khỏi manifest, giữ bản ở train, danh sách nằm trong `dropped_exact_duplicates` của audit.json, file gốc không bị xóa; ảnh gần trùng được liệt kê để rà soát. dHash là công cụ sàng lọc, không chứng minh hai ảnh độc lập.

Audit hiện hai thanh tiến trình: `[1/2] Images + labels` đếm ảnh và `[2/2] Duplicate scan` đếm các cặp đã duyệt (gồm cả cặp cùng split được bỏ qua), kèm số cặp trùng/gần trùng. Thanh hiển thị thời gian, tốc độ và ETA qua stderr để dùng được khi gọi bằng subprocess trong Kaggle. CLI chỉ in tóm tắt cuối; danh sách chi tiết nằm trong audit.json để tránh làm đầy output notebook.

```bash
python audit.py /path/to/datasets --output /kaggle/working/prepared
# Khi có metadata nguồn thật:
python audit.py /path/to/datasets --output /kaggle/working/prepared --groups groups.csv
python train.py --data /kaggle/working/prepared/data.yaml --output /kaggle/working/experiments --name underwater --resume
# Roboflow export 2 lớp:
python audit.py /path/to/abovewater --output /kaggle/working/prepared/abovewater --drop-exact-duplicates
python train.py --data /kaggle/working/prepared/abovewater/data.yaml --output /kaggle/working/experiments/abovewater --name abovewater --resume
```

CSV có cột `image,group`, ví dụ `train/frame_00001.jpg,recording_A`. Không suy ra recording chỉ từ số frame. Có nhóm nguồn thì chia gần 70/15/15 theo nhóm, seed 42; số ảnh có thể lệch tỷ lệ để không cắt nhóm. Không có nhóm thì giữ split gốc, không tạo test giả từ các frame có nguy cơ cùng cảnh. Nếu audit phát hiện duplicate, sửa dataset trong bản sao rồi chạy lại; giữ một bản ảnh trùng ở cùng một split, không xóa âm thầm dữ liệu gốc.

Train YOLO11n và YOLO11s, imgsz640, tối đa100 epoch, patience15, seed42, batch16→8→4 khi CUDA OOM. Resume từ last.pt; nếu OOM lúc resume thì bắt đầu lại thí nghiệm đó với batch thấp hơn. Lưu checkpoint mỗi5 epoch và bản last/best. Chỉ resume khi manifest dữ liệu không đổi. Khi chuyển Kaggle session, attach output version trước và copy experiments vào working; lưu các thư viện và phần cứng trong environment.json.

Chọn bằng validation: giữ ứng viên có recall drowning cách tốt nhất ≤1 điểm phần trăm; tiếp tục giữ precision cách tốt nhất trong nhóm ≤1 điểm; chọn latency p50 thấp nhất. Đây là quy tắc cố định, không dùng kết quả test để đổi lựa chọn. Có test độc lập thì chạy một lần với `--test-checkpoint` trỏ checkpoint đã chọn; notebook tự làm việc này cho góc nhìn có test (Roboflow). Không có test (Figshare) thì ghi rõ chỉ có validation. Test của Roboflow chia ngẫu nhiên theo ảnh, có thể chứa frame cùng cảnh với train, nên vẫn cần đánh giá trên video độc lập.

Tải `results.zip` ở mục Output: mỗi góc nhìn có `<view>_selected.pt`, selected/comparison/test/environment JSON, manifest, audit, plots (bỏ checkpoint epoch và last.pt). Chép `underwater_selected.pt` và `abovewater_selected.pt` vào `fastapi-backend/model/detection_model/`: backend tự dùng các file này, tự quay về `underwaterModel.pt`/`abovewaterModel.pt` khi file bị xóa. `UNDERWATER_MODEL`/`ABOVE_MODEL` vẫn ghi đè được đường dẫn. Lớp struggling và cảnh báo vàng `struggling_alert` chỉ có ở camera dưới nước; camera trên mặt nước phân biệt swimming/drowning và vật nổi `float`. Backend vẽ `float` lên khung hình (màu xanh) nhưng không đưa vào DeepSORT, danh sách người hay bộ cảnh báo.

## Đánh giá và nghiệm thu

```powershell
# Trong fastapi-backend:
python -m unittest discover -s tests -v
python tools/benchmark.py --video ../frontend/public/demo_video/ZoneA_Underwater.mp4 --frames 100 --output benchmark.json
# Cần môi trường GAN đã cài để so sánh:
python tools/benchmark.py --video ../frontend/public/demo_video/ZoneA_Underwater.mp4 --frames 100 --compare-gan --output benchmark-gan.json
# Nguồn phải chạy đủ 30 phút, thường là webcam:
python tools/soak.py --session SESSION_ID --seconds 1800 --output soak-report.json
```

Trong root repo: `python -m unittest discover -s model_training/pipeline -p test_pipeline.py -v`. Trong frontend: `npm test`, `npm run lint`, `npm run build` (bao gồm TypeScript).

Backend ghi `processed_outputs/{session_id}.jsonl`. Để chấm sự kiện, gán nhãn video độc lập với JSON `{video_id, duration_seconds, events:[{track_id,start,end}]}`; track_id phải được đối chiếu thủ công với đúng người trên video. Các mốc start/end dùng giây của video. Không chấm cùng dữ liệu đã dùng chọn ngưỡng.

```bash
python model_training/pipeline/evaluate_events.py --truth truth.json --predictions fastapi-backend/processed_outputs/SESSION_ID.jsonl --output events-report.json
```

Chấm theo episode cảnh báo đỏ, không đếm pulse còi lặp. Báo cáo cảnh báo sai/phút, tỷ lệ bỏ sót, độ trễ; track đổi ID cần được đối chiếu trước khi chấm. Soak xuất mẫu RAM/FPS/p95 và mức tăng RAM, kết thúc sớm là fail. `passed` của soak chỉ xác nhận nguồn tồn tại đủ thời gian; cần xem xu hướng RAM và độ trễ để nghiệm thu ổn định.

Không công bố mAP, FPS mục tiêu, checkpoint mới hay kết quả soak 30 phút cho đến khi thực sự chạy các bước tương ứng. Kết quả kiểm tra triển khai và các giới hạn được ghi trong [VALIDATION.md](VALIDATION.md).
