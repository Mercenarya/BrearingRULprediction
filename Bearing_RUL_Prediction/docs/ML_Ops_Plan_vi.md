# Kế hoạch và Quy trình ML Ops cho Dự án Dự đoán RUL của Bearing

## 1. Quản lý Dữ liệu

### 1.1 Lưu trữ Dữ liệu

- Lưu trữ các tập dữ liệu đã chuẩn hoá trong thư mục `data/processed/`.
- Các tập dữ liệu hiện tại:
  - `train.npz`, `val.npz`, `test.npz`: Tập dữ liệu chính (có thể là XJTU-SY)
  - `train_pronostia.npz`, `val_pronostia.npz`, `test_pronostia.npz`: Tập dữ liệu Pronostia
  - `train_xjtusy.npz`, `val_xjtusy.npz`, `test_xjtusy.npz`: Tập dữ liệu XJTU-SY (phân chia thay thế)
- Mỗi file `.npz` chứa ba mảng:
  - `vibration`: Hình dạng `(n_samples, window_size, 2)` (float32)
  - `temperature`: Hình dạng `(n_samples, window_size, 1)` (float32)
  - `rul`: Hình dạng `(n_samples,)` (float32)

### 1.2 Phiên bản Dữ liệu

- Duy trì file `data/registry.csv` (hoặc JSON) để theo dõi:
  - Tên tập dữ liệu
  - Nguồn (ví dụ: XJTU-SY, Pronostia)
  - Phiên bản/ngày
  - Các bước tiền xử lý đã áp dụng (lọc cho tần số, giảm mẫu, chuẩn hoá)
  - Tổng kiểm tra file (MD5 hoặc SHA256) để xác minh tính toàn vẹn
- Cập nhật registry khi có tập dữ liệu mới được thêm vào.

### 1.3 Kiểm tra Dữ liệu

- Triển khai script kiểm tra để xác minh:
  - Tồn tại và khả năng đọc file
  - Các khóa bắt buộc (`vibration`, `temperature`, `rul`)
  - Kiểu dữ liệu (float32)
  - Độ nhất quán hình dạng (kích thước cửa sổ 1024, kênh như mong đợi)
  - Phạm vi giá trị (không có NaN hoặc Inf)

## 2. Thiết lập Pipeline

### 2.1 Pipeline Huấn luyện

- Sử dụng script hiện có: `src/training/train.py`
- Cấu hình qua `config.yaml`:
  - Tham số dữ liệu: sampling_rate, window_size, overlap, normalization
  - Tham số mô hình: thang đo CNN, loại fusion, kích thước LSTM, v.v.
  - Tham số huấn luyện: kích thước batch,学习率, số epoch, optimizer
- Script tự động:
  - Tải dữ liệu từ `data/processed/train.npz`, `val.npz`, `test.npz`
  - Tạo DataLoader qua `src/data/dataset.py`
  - Huấn luyện mô hình với ghi nhật ký Weights & Biases (nếu được bật)
  - Lưu mô hình tốt nhất vào `checkpoints/best_model.pt`
  - Đánh giá trên tập test và ghi lại metrics

### 2.2 Pipeline Đánh giá

- Mở rộng script đánh giá hiện có (`src/evaluation/evaluate.py`) để hỗ trợ đánh giá xuyên bộ dữ liệu.
- Tạo script mới `src/evaluation/cross_dataset_evaluate.py` mà:
  - Tải checkpoint mô hình đã huấn luyện
  - Đánh giá trên nhiều bộ dữ liệu (chính, Pronostia, XJTU-SY)
  - Tính toán và so sánh các metrics giữa các bộ dữ liệu
  - Tạo báo cáo (bảng/đồ thị) montrant sự suy giảm/tăng trưởng hiệu suất

### 2.3 Pipeline Tiền xử lý (nếu sử dụng dữ liệu thô)

- mặc dù chúng ta đang sử dụng dữ liệu đã chuẩn hoá, pipeline tiền xử lý dữ liệu thô có sẵn trong:
  - `src/data/download.py`: Tải xuống từ Kaggle
  - `src/data/preprocessing.py`: Áp dụng bộ lọc cho tần số, giảm mẫu, chuẩn hoá
  - `src/data/sliding_window.py`: Tạo các cửa sổ trượt (nếu cần)
- Nếu có dữ liệu thô, thực hiện:
  ```bash
  python src/data/download.py
  python src/data/preprocessing.py
  ```
- Đầu ra nên được đặt trong `data/processed/` với cấu trúc `.npz` tương tự.

## 3. Theo dõi Thử nghiệm

### 3.1 Tích hợp Weights & Biases

- Bật trong `config.yaml`: `logging.use_wandb: true`
- Cấu hình tên dự án: `logging.project_name`
- Mỗi lần chạy huấn luyện ghi lại:
  - Siêu tham số (từ config)
  - Các metrics mỗi epoch (loss, MAE, RMSE, v.v.)
  - Mô hình artifact (checkpoints)
  - Các metrics hệ thống (sử dụng GPU, v.v.)
- S용 dụng bảng điều khiển W&B để so sánh các lần試 nghiệm.

### 3.2 Khả năng tái tạo Thử nghiệm

- Gắn thẻ cho mỗi lần試 nghiệm bằng:
  - Mã commit Git
  - Phiên bản tập dữ liệu (từ registry)
  - Cấu hình tiền xử lý
- Lưu trữ mô hình cuối cùng và metadata dưới dạng artifact trong W&B.

## 4. Quản lý Phiên bản Mô hình

### 4.1 Quản lý Checkpoint

- Script huấn luyện lưu trữ:
  - `checkpoints/last_model.pt`: Epoch gần nhất
  - `checkpoints/best_model.pt`: Loss validation tốt nhất
- Bao gồm trong checkpoint:
  - Trạng thái dict mô hình
  - Trạng thái dict optimizer
  - Epoch
  - Loss validation
  - Cấu hình được sử dụng

### 4.2 Registry Mô hình (Tương lai)

- Dành cho sản xuất, cân nhắc sử dụng MLflow hoặc W&B Model Registry để:
  - Giai đoạn mô hình (Staging, Production)
  - Theo dõi bloodline mô hình
  - Cho phép quay lại mô hình

## 5. Chuẩn bị Triển khai

### 5.1 Xuất Mô hình

- Xuất mô hình đã huấn luyện sang TorchScript hoặc ONNX để phục vụ:
  ```python
  # Ví dụ xuất TorchScript
  model.eval()
  traced_model = torch.jit.trace(model, (vib_example, temp_example))
  traced_model.save("model_traced.pt")
  ```

### 5.2 Mẫu Dịch vụ Dự đoán

- Tạo một dịch vụ suy luận đơn giản (ví dụ: FastAPI) mà:
  - Tải mô hình đã xuất
  - Chấp nhận các mảng vibration và temperature
  - Trả về dự đoán RUL
  - Bao gồm xác thực đầu vào và xử lý lỗi

### 5.3 Quản lý Phụ thuộc

- Xuất các yêu cầu cho triển khai:
  ```bash
  pip freeze > requirements_deployment.txt
  ```
- Đảm bảo tính tương thích với môi trường đích (CPU/GPU, phiên bản Python).

## 6. Giám sát và Bảo trì

### 6.1 Giám sát Huấn luyện

- Giám sát nhật ký huấn luyện để tìm:
  - Sự hội tụ của loss
  - Vượt quá fitting (train vs. val loss)
  - Các bất thường (loss NaN, gradient nổ)
- Thiết lập警報 (qua W&B hoặc tùy chỉnh) cho những lỗi huấn luyện.

### 6.2 Giám sát Mô hình (Sau khi triển khai)

- Theo dõi độ lệch dự đoán:
  - So sánh phân phối RUL dự đoán với mong đợi
  - Giám sát thống kê dữ liệu đầu vào (vibration, temperature) để tìm sự dịch chuyển
- Lên lịch reassessment định kỳ trên dữ liệu mới nhất.

### 6.3 Giám sát Pipeline Dữ liệu

- Xác thực schema và chất lượng dữ liệu vào來
- Báo警 về dữ liệu thiếu hoặc lỗi tiền xử lý

## 7. Quy trình làm việc

### Bước 1: Chuẩn bị Dữ liệu

1. Sao chép các file `.npz` đã chuẩn hoá vào `data/processed/`:
   ```bash
   cp /path/to/dataset/*.npz E:/BearingRULprediction/Bearing_RUL_Prediction/data/processed/
   ```
2. Chạy script kiểm tra dữ liệu (sẽ được tạo) để xác minh tính toàn vẹn.
3. Cập nhật `data/registry.csv` với thông tin về tập dữ liệu.

### Bước 2: Cấu hình

1. Xem lại và điều chỉnh `config.yaml}\) khi cần (ví dụ: kích thước batch, learning rate).
2. Đảm bảo `logging.use_wandb: true` để theo dõi thử nghiệm.
3. Đặt `data.normalization` để khớp với tiền xử lý đã áp dụng (nên là "standard").

### Bước 3: Huấn luyện

1. Thực hiện huấn luyện:
   ```bash
   python src/training/train.py
   ```
2. Theo dõi tiến độ qua:
   - Đầu ra console
   - Bảng điều khiển Weights & Biases (nếu được bật)
3. Đầu ra huấn luyện:
   - Các checkpoint trong thư mục `checkpoints/`
   - Metrics cuối cùng được ghi log vào W&B và console

### Bước 4: Đánh giá

1. Đánh giá trên tập test chính (đã được thực hiện trong script huấn luyện).
2. Chạy đánh giá xuyên bộ dữ liệu:
   ```bash
   python src/evaluation/cross_dataset_evaluate.py --checkpoint checkpoints/best_model.pt
   ```
3. Phân tích kết quả:
   - Tập dữ liệu chính: Hiệu suất như mong đợi
   - Tập dữ liệu xuyên bộ: Kiểm tra khoảng cách tổng quát hoá

### Bước 5: Lựa chọn và Thúc đẩy Mô hình

1. Chọn mô hình tốt nhất dựa trên:
   - Hiệu suất validation/test trên tập dữ liệu chính
   - Độ bền xuyên bộ dữ liệu (nếu yêu cầu)
2. Đẩy mô hình vào sản xuất:
   - Sao chép `checkpoints/best_model.pt` vào `models/production/`
   - Xuất sang TorchScript/ONNX để triển khai
   - Cập nhật модель registry (nếu có)

### Bước 6: Tài liệu và Báo cáo

1. Cập nhật `README.md` với:
   - Hướng dẫn để tái tạo kết quả
   - Mô tả tập dữ liệu và nguồn
   - Tóm tắt hiệu suất mô hình
2. Tạo báo cáo (PDF/HTML) tóm tắt:
   - Các lần试 nghiệm đã thực hiện
   - Hiệu suất mô hình tốt nhất
   - Phân tích so sánh giữa các bộ dữ liệu
   - Những cân nhắc cho việc triển khai

### Bước 7: Bảo trì

1. Lên lịch huấn luyện lại định kỳ (ví dụ: hàng tháng) khi có dữ liệu mới đến.
2. Giám sát mô hình sản xuất để tìm dấu hiệu lệch dữ liệu và suy giảm hiệu suất.
3. Cập nhật registry tập dữ liệu và các bước tiền xử lý nếu cần.

## 8. Công cụ và Công nghệ

- **Framework**: PyTorch
- **Theo dõi Thử nghiệm**: Weights & Biases (đã tích hợp)
- **Kiểm tra Dữ liệu**: Script tùy chỉnh (sử dụng NumPy)
- **Xuất Mô hình**: TorchScript, ONNX
- **Triển khai**: FastAPI/Docker (mẫu sẽ được tạo)
- **Quản lý Phiên bản**: Git (đối với mã), DVC hoặc MLflow (đề xuất cho dữ liệu/mô hình)

## 9. Các Bước Tiếp theo

1. Triển khai script kiểm tra dữ liệu.
2. Tạo script đánh giá xuyên bộ dữ liệu.
3. Tạo mẫu xuất mô hình và giao diện suy luận.
4. Thiết lập pipeline huấn luyện lại tự động (ví dụ: bằng GitHub Actions hoặc cron).
5. Ghi chú toàn bộ quy trình trong thư mục `docs/` của dự án.

---

*Kế hoạch được tạo ngày: 2026-09-30*
*Đối với: Vai trò ML Ops - Dự án Dự đoán RUL của Bearing*
