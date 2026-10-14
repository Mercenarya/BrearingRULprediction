# Chạy hai notebook trên Google Colab

## 1. Chuẩn bị

Nhận `handover_notebook_only.zip` mới nhất. Trên https://colab.research.google.com/ đăng nhập Google, chọn Upload notebook và mở **data_process.ipynb**. Runtime CPU là đủ cho preprocessing; không cần GPU.

Chạy Run all. Các cell đầu nhận diện Colab, cấu hình đường dẫn, nhận ZIP bàn giao và cài dependency bằng `python -m pip install -r requirements.txt`. Khi hộp upload hiện ra, chọn **handover_notebook_only.zip**, không chọn raw dataset ZIP. Cell chỉ giải nén notebook/config/dependencies; reports Windows trong ZIP không được dùng làm bằng chứng Colab.

Sau khi data notebook chạy xong 38 tests, tải notebook `.ipynb` có output bằng File → Download. Tiếp theo Upload notebook **data_pipeline_scripts.ipynb**, chọn các đường dẫn/mode ở đầu và Run all. Nếu Colab cấp runtime khác, upload lại ZIP khi được yêu cầu.

## 2. Chạy tại /content, tự tải raw

Giữ cấu hình mặc định:

```python
COLAB_MOUNT_DRIVE = False
COLAB_WORK_DIR = '/content/bearing_rul/handover_notebook_only'
COLAB_RAW_SOURCE = 'download'
COLAB_RAW_ROOT = '/content/datasets/pronostia/ieee-phm-2012-data-challenge-dataset-master'
COLAB_OUTPUT_BASE = '/content/bearing_rul_outputs'
```

Pipeline tải archive từ URL trong YAML, kiểm **SHA256 và ZIP commit comment**, rồi giải nén vào RAW_ROOT mới. Nếu snapshot master đã đổi, checksum fail và pipeline dừng; không tự nhận snapshot khác. Cần đủ Learning_set, Test_set, Full_Test_Set. Đường dẫn RAW_ROOT là folder chứa trực tiếp ba tập này.

OUTPUT_ROOT được tạo thành `COLAB_OUTPUT_BASE/<timestamp>_<random_id>`; clean_rebuild không ghi đè/xóa cache cũ. `max_files=None`, rebuild toàn bộ raw.

## 3. Raw ZIP trên Google Drive

Upload ZIP raw **phm2012-master.zip** vào `My Drive/Bearing_RUL/`, rồi đặt:

```python
COLAB_MOUNT_DRIVE = True
COLAB_RAW_SOURCE = 'drive_zip'
COLAB_DRIVE_ARCHIVE = '/content/drive/MyDrive/Bearing_RUL/phm2012-master.zip'
COLAB_RAW_ROOT = '/content/datasets/pronostia/ieee-phm-2012-data-challenge-dataset-master'
```

Bạn tự xác nhận hộp cấp quyền của `drive.mount('/content/drive')`. Pipeline đọc một archive từ Drive và giải nén về /content để hạn chế I/O nhiều file nhỏ. Archive phải khớp checksum đã pin trong YAML.

## 4. Raw đã giải nén và output trên Drive

```python
COLAB_MOUNT_DRIVE = True
COLAB_RAW_SOURCE = 'existing'
COLAB_RAW_ROOT = '/content/drive/MyDrive/Bearing_RUL/raw/ieee-phm-2012-data-challenge-dataset-master'
COLAB_OUTPUT_BASE = '/content/drive/MyDrive/Bearing_RUL/outputs'
```

Có thể đặt COLAB_WORK_DIR tới folder trên Drive có đủ bộ notebook/config. Chạy trực tiếp hàng chục nghìn CSV trên Drive thường chậm hơn /content. Nếu xử lý ở /content, tải output/report trước khi xóa runtime; các file VM không tự được lưu cùng notebook.

## 5. Bằng chứng execution Colab

Notebook data tạo `reports/data_notebook_colab_execution.json`: 38 tests, Python/package versions và platform. Notebook pipeline tạo:

- `OUTPUT_ROOT/reports/colab_execution.json`: runtime google_colab, timestamps UTC, raw/output paths, nguồn/checksum, tests, full-pass validation, shapes, ground truth warnings và hash notebook nguồn.
- `clean_rebuild_execution.json`: rebuilt=true, preprocessing PASS, 6 development folds và official split.
- Các report quality, labels, temperature coverage, QC và validation của lượt chạy đó.
- `OUTPUT_ROOT/colab_reports.zip`: chỉ chứa reports mới; cell cuối tải ZIP về máy.

Report Colab **chỉ được viết khi code thực thi ở runtime nhận diện Google Colab**. Windows/Linux local không được ghi thành Colab PASS. Report mới không dùng các output Windows trong ZIP làm bằng chứng.

Nếu hai notebook dùng VM khác nhau, tải cả report data notebook và report pipeline để gửi cùng hai notebook có output; report data notebook nằm trong Files → `/content/bearing_rul/handover_notebook_only/reports/`. Nếu chung VM, pipeline tự chép report data vào reports ZIP.

Giữ cảnh báo Bearing1_4: nhãn 339 s, terminal-minus-cut 2890 s; official_benchmark_ground_truth_ready=false. Không công bố metric official.

Nguồn hướng dẫn Colab: [FAQ chính thức](https://research.google.com/colaboratory/faq.html), mục Using Colab/Google Drive I/O.
