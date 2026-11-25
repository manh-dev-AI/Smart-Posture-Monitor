# Tài liệu Kiến trúc Kỹ thuật — Smart Posture Monitor (V03)

Tài liệu này tóm tắt thiết kế pipeline, biểu diễn đặc trưng, cơ chế ổn định thời gian thực, giao thức WebSocket và kết quả đánh giá của mô hình đang dùng trong runtime.

---

## 1. Camera Protocol & Pose Estimation (`src/pose_detector.py`)

- **Góc đặt camera:** Thiết kế cho camera đặt lệch khoảng **45° từ bên trái** người dùng. Không lật ngược ảnh (mirror) trước khi inference vì thao tác này làm đảo ngược nhãn `lean_left` và `lean_right`.
- **6 Keypoints sử dụng (từ COCO 17 điểm):**
  `nose` (0), `left_eye` (1), `right_eye` (2), `left_ear` (3), `left_shoulder` (5), `right_shoulder` (6).
  Điểm `right_ear` (4) bị loại bỏ do thường xuyên bị che khuất ở góc nghiêng 45°.
- **Chọn chủ thể chính:** Khi có nhiều người trong khung hình, `PoseDetector` chấm điểm theo công thức `0.7 * centrality_score + 0.3 * confidence_score` để ưu tiên người ngồi ở trung tâm.
- **Tương thích phần cứng:** `_resolve_device()` tự động kiểm tra khả năng thực thi kernel CUDA và tự fallback về `cpu` nếu GPU/PyTorch không tương thích, tránh crash khi khởi động.

---

## 2. Biểu diễn đặc trưng: RAW12 → Personal Calibration → REP13

### 2.1. RAW12 (`src/feature_extractor.py`)

Mỗi frame có đủ 6 keypoints với `confidence >= 0.35` được trích xuất thành vector 12 chiều (lấy từ `FeatureExtractor.FEATURE_NAMES`):

| Nhóm | Đặc trưng |
|---|---|
| Angle / Orientation | `shoulder_roll_deg`, `eye_roll_deg`, `head_shoulder_roll_diff_deg` |
| Sagittal / Neck | `neck_pitch_deg` (vector `left_ear → nose`) |
| Spatial (pixel) | `eye_center_x_px`, `eye_center_y_px`, `shoulder_center_x_px`, `eye_shoulder_vertical_gap_px`, `eye_shoulder_horizontal_offset_px` |
| Perspective / Scale | `inter_eye_distance_px`, `ear_nose_horizontal_span_px`, `shoulder_width_px` |

### 2.2. Personal Calibration (`src/personal_calibration.py`)

- Thu thập **30 frame** hợp lệ ở tư thế ngồi thẳng (`correct`).
- Các đặc trưng phi góc lấy **median** thông thường.
- Các đặc trưng góc xử lý bằng **circular median-like** (`_circular_median_deg`) để tránh sai lệch tại biên $\pm 180^\circ$.
- Khi reset phiên (`reset_session()`), baseline cá nhân vẫn được giữ nguyên; chỉ reset khi người dùng chủ động hiệu chuẩn lại (`start_calibration()` / `reset_calibration()`).

### 2.3. REP13 (`src/representation_builder.py`)

Từ vector hiện tại `RAW12` và `baseline RAW12`, hệ thống biến đổi stateless sang 13 đặc trưng (`RepresentationBuilder.OUTPUT_FEATURE_NAMES`):

1. **4 Wrapped Angle Deltas:** độ lệch góc theo chu kỳ $[-180^\circ, 180^\circ)$ cho 4 góc RAW.
2. **5 Body-Normalized Spatial Deltas:** $(f_{\text{curr}} - f_{\text{base}}) / \text{shoulder\_width}_{\text{base}}$.
3. **3 Scale Log-Ratios:** $\ln(f_{\text{curr}} / f_{\text{base}})$ cho 3 đặc trưng phối cảnh/khoảng cách.
4. **1 Head Drift Magnitude:** $\sqrt{\Delta x_{\text{eye\_body}}^2 + \Delta y_{\text{eye\_body}}^2}$.

Theo đo đạc trên tập dữ liệu V03, bước chuyển sang REP13 giúp giảm median normalized subject spread từ `0.6321` (RAW12) xuống `0.4185` (REP13).

---

## 3. Hậu xử lý & Thống kê thời gian thực

### 3.1. Temporal Smoothing & Yaw Gating (`src/temporal_monitor.py`)

- **Cấu hình runtime mặc định** (trong `PostureInferenceEngine`):
  - `window_size = 7`
  - `min_samples = 3`
  - `hysteresis_frames = 2`
  - `yaw_mode = "conservative"` (`DEFAULT_YAW_LOW = 0.45`, `DEFAULT_YAW_HIGH = 1.80`)
- **Yaw Gating:** Dùng tỷ số `face_rotation_proxy = dist(left_eye, nose) / dist(right_eye, nose)`. Khi người dùng quay đầu vượt ngưỡng, chế độ `conservative` giữ nguyên nhãn ổn định gần nhất và không đưa frame nhiễu vào cửa sổ vote.
- **Lọc trạng thái điều khiển:** Các trạng thái `no_person`, `evaluating`, `head_turned`, `low_confidence` và các trạng thái calibration không bao giờ đi vào `deque` majority vote.

### 3.2. Session Statistics (`src/session_statistics.py`)

- Tích lũy `correct_seconds` và `bad_seconds` dựa trên `time.monotonic()` thay vì giả định FPS cố định.
- **Cảnh báo theo episode:** Một chuỗi tư thế sai liên tục vượt ngưỡng `alert_threshold_seconds` (mặc định `2.0s`, chỉnh được từ 1–60s qua API) chỉ ghi nhận **1 violation**; episode chỉ reset khi trở lại `correct` hoặc gặp trạng thái trung gian (`NO_PERSON`, `LOW_CONFIDENCE`, calibration).
- `ergonomics_score`: Tính theo phần trăm thời gian ngồi đúng trên tổng thời gian hợp lệ (`correct_seconds / (correct_seconds + bad_seconds) * 100`).

---

## 4. Runtime API & WebSocket (`app/app.py`, `src/inference.py`)

### Endpoints

- `GET /`: Trả giao diện Dashboard ([app/templates/index.html](../app/templates/index.html)).
- `GET /api/info`: Trả tên model, YOLO weights, số feature, danh sách lớp và trạng thái calibration.
- `POST /api/calibrate`: Kích hoạt quy trình hiệu chuẩn 30 mẫu.
- `POST /api/reset_session`: Đặt lại thống kê và bộ đệm temporal (giữ nguyên baseline cá nhân).
- `POST /api/set_alert_threshold`: Cập nhật ngưỡng thời gian cảnh báo (`1–60` giây).
- `WS /ws/stream`: Nhận payload JSON chứa `action` (`calibrate`, `reset_calibration`) hoặc ảnh JPEG base64 + `client_time`, trả về kết quả phân tích kèm `server_processing_ms`.

### Cơ chế chống backlog

Frontend giữ cờ `frameInFlight` để đảm bảo **tại một thời điểm chỉ có tối đa 1 frame đang suy luận trên server** (*one-frame-in-flight / latest-frame-wins*). Luồng hiển thị webcam và vẽ skeleton chạy qua `requestAnimationFrame()` độc lập với tốc độ suy luận AI.

### Các trạng thái `status` của `PostureInferenceEngine`

| Status | Ý nghĩa |
|---|---|
| `NO_PERSON` | Không phát hiện người trong khung hình |
| `LOW_CONFIDENCE` | Có người nhưng thiếu keypoint hoặc confidence dưới ngưỡng `0.35` |
| `CALIBRATION_REQUIRED` | Chưa có personal baseline |
| `CALIBRATING` | Đang thu thập 30 mẫu `correct` |
| `CALIBRATION_SAMPLE_REJECTED` | Mẫu calibration lỗi hoặc calibration đã hoàn thành |
| `CALIBRATED` | Vừa hoàn thành đủ 30 mẫu baseline |
| `OK` | Đã dự đoán và làm mượt tư thế thành công |

---

## 5. Tham chiếu kết quả đánh giá mô hình

Số liệu trích từ `models/training_metadata.json` (Nested LOPO trên 14 người / 16 phiên ghi, 3,976 mẫu REP13):

### So sánh các mô hình ứng viên

| Model | LOPO Macro F1 Mean | Std | Worst-person F1 | LOPO Accuracy Mean | Correct F1 |
|---|---:|---:|---:|---:|---:|
| **SVM_RBF** | **0.919948** | **0.110044** | **0.671906** | **0.932851** | **0.938104** |
| Random_Forest | 0.899258 | 0.137486 | 0.644433 | 0.915897 | 0.914313 |
| MLP | 0.899089 | 0.123974 | 0.618682 | 0.917051 | 0.939971 |
| XGBoost | 0.873577 | 0.146456 | 0.631538 | 0.894725 | 0.909948 |

### Hiệu năng từng lớp của SVM RBF

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| `correct` | 0.8973 | 0.9828 | 0.9381 | 987 |
| `forward_slouch` | 0.8960 | 0.9200 | 0.9078 | 937 |
| `lean_left` | 0.9841 | 0.9648 | 0.9744 | 1024 |
| `lean_right` | 0.9742 | 0.8804 | 0.9249 | 1028 |
