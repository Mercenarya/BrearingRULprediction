# Báo Cáo Test Train - Dự đoán RUL của Bearing

## Tổng quan
Ngày 13/10/2026, đã thực hiện quá trình test train cho mô hình dự đoán Remaining Useful Life (RUL) của bearing dựa trên dữ liệu đa modal (vibrations và temperature).

## Các bước đã thực hiện

### 1. Đọc và nghiên cứu dữ liệu handover
- Đã kiểm tra thư mục `data/raw/customer_handover/customer_handover/`
- Tài liệu chính bao gồm:
  - `README.md`: Hướng dẫn về preprocessing dữ liệu PRONOSTIA
  - `data_process.ipynb`: Notebook xử lý dữ liệu chính
  - `data_pipeline_scripts.ipynb`: Script pipeline để tái tạo dữ liệu từraw
  - Cấu hình YAML và file ground truth

### 2. Xác thực dữ liệu
- Đã chạy script `src/data/validate_data.py`
- Kết quả: Tất cả 9 file .npz trong `data/processed/` đều hợp lệ
  - test.npz, test_pronostia.npz, test_xjtusy.npz
  - train.npz, train_pronostia.npz, train_xjtusy.npz
  - val.npz, val_pronostia.npz, val_xjtusy.npz
- Mỗi file có:
  - vibration: shape (20400, 1024, 2), dtype float32
  - temperature: shape (20400, 1024, 1), dtype float32
  - rul: shape (20400,), dtype float32

### 3. Tạo registry
- Đã chạy script `src/data/generate_registry.py`
- Tạo file `data/processed/registry.csv` chứa metadata cho tất cả file .npz
- Mỗi entry bao gồm: filename, source (Pronostia/XJTU-SY/Unknown), version, preprocessing_steps, SHA256 hash, size_bytes, date_added

### 4. Cấu hình training
- Đã sửa đổi `src/training/train.py` và `src/training/trainer.py` để loại bỏ wandb (do lỗi protobuf)
- Config.yaml được настроен với:
  - num_epochs: 2 (được giảm để test nhanh)
  - batch_size: 20
  - learning_rate: 0.0001
  - loss_function: stage_aware_phm_loss
  - Kích thước window: 1024 mẫu
  - Overlap: 50%

### 5. Thực thi test train
- Đã chạy lệnh: `python src/training/train.py`
- Training thành công hoàn thành 2 epochs
- Tạo checkpoint:
  - checkpoint_epoch_0.pth
  - checkpoint_epoch_1.pth
  - best_model.pt (mô hình tốt nhất dựa trên validation loss)

## Kết quả
- Pipeline dữ liệu đã được xác thực thành công
- Mô hình có thểtrain mà không gặp lỗi
- Checkpoints đã được lưu thành công
- Chuẩn bị sẵn sàng để thực hiện training đầy đủ hoặc cross-dataset evaluation

## Đề xuất для các bước tiếp theo
1. Chạy training đầy đủ với số epochs tăng lên (ví dụ: 50-100)
2. Thực hiện cross-dataset evaluation bằng script `src/evaluation/cross_dataset_evaluate.py`
3. Phân tích kết quả và tinh chỉnh siêu tham số
4. Tối ưu hóa cấu trúc mô hình nếu cần thiết

## Lưu ý تقنية
- Loại bỏ wandb do xung đột phiên bản protobuf trong môi trường Windows
- Sử dụng CPU cho training (không có GPU khả dụng trong môi trường này)
- Verlust hàm trong giai đoạn train nhìn hợp lý (dao động quanh 0, cho thấy mô hình đang học)

---
*Báo cáo này được viết bằng tiếng Việt theo yêu cầu*