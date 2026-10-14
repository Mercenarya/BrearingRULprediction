# Dữ liệu từ lần chạy notebook mới nhất

**Đã chạy xong hai notebook, 38 tests PASS, 6 development folds và official DataLoader PASS. Clean rebuild: true. Không có lỗi cell hoặc lỗi validation.**

Thư mục dữ liệu: [pronostia_feedback_clean_20261006_174357](C:/Users/ACER/Downloads/Bearing_RUL_Prediction/artifacts/pronostia_feedback_clean_20261006_174357)

Dung lượng thực tế: **770.2 MiB**, 123 files; gồm **70 shard NPZ**, 17 shard acquisition cho 17 bearings. Raw vẫn nằm ngoài project: `C:\Users\ACER\Downloads\PHM_2012_Raw\ieee-phm-2012-data-challenge-dataset-master`.

## Dùng khi train

| Đường dẫn trong thư mục output | Nội dung |
|---|---|
| fold_0/ đến fold_5/ | 6 development folds: train/validation, scaler riêng fit trên train |
| official/shards/train/ | Dữ liệu đã scale của 6 learning bearings |
| official/shards/test/ | Dữ liệu đã scale của 11 test bearings; giữ cảnh báo ground truth |
| acquisitions/ | 17 shard trước bước scale theo fold |
| Mỗi fold: cache_index.csv / cache_index.parquet | Chỉ mục window, split, bearing và đường dẫn shard |
| Mỗi fold: scaler.json / split.json / cache_identity.json | Thống kê train, split và provenance |
| manifest.csv / manifest_all_files.csv / preprocessing.yaml | Truy vết raw, acquisition, nhãn, temperature alignment và cấu hình |

Mỗi sample từ Dataset có vibration **[32,2,1024]**, temperature và temperature_mask **[32,1,64]**, RUL **[1]**. DataLoader batch_size=32 cho vibration **[32,32,2,1024]**, temperature/mask **[32,32,1,64]**, RUL **[32,1]**. Mask có dtype bool, True biểu thị thiếu; giá trị thiếu bằng 0 sau scale.

Không cần tự ghép NPZ thành window: dùng **PronostiaDataset / pronostia_loader** đã có trong data_process.ipynb; artifact_dir trỏ vào fold_0…fold_5 hoặc official, split là train/validation/test theo split.json. Toàn bộ các batch đã được đọc full-pass, kiểm finite và missing mask.

## Số window đã đọc full-pass

| Fold | Split | Windows |
|---|---|---:|
| 0 | train | 288 |
| 0 | validation | 175 |
| 1 | train | 409 |
| 1 | validation | 54 |
| 2 | train | 407 |
| 2 | validation | 56 |
| 3 | train | 414 |
| 3 | validation | 49 |
| 4 | train | 431 |
| 4 | validation | 32 |
| 5 | train | 361 |
| 5 | validation | 102 |
| official | train | 462 |
| official | test | 866 |

## Giới hạn sử dụng

Bearing1_4 giữ nhãn **339 s** theo bảng BTC; mismatch với Full_Test_Set **2890 s** vẫn được cảnh báo. `official_benchmark_ground_truth_ready=false`: dùng development folds để phát triển/kiểm tra mô hình; chưa công bố metric official trên test set. Xem ROLE1_SECTION5_ACCEPTANCE.md.

## Báo cáo và notebook

Notebook có output mới: handover_notebook_only/data_process.ipynb và data_pipeline_scripts.ipynb; đã đồng bộ bản sao trong src/data và scripts. Reports của lượt chạy mới ở handover_notebook_only/reports/. clean_rebuild_execution.json và notebook_execution.json là bằng chứng thực thi, validation_report.json là full-pass check.

ZIP handover_notebook_only.zip chứa notebook/config/nguồn nhãn/báo cáo; **không chứa cache lớn**. Muốn gửi dữ liệu tensor cho leader thì gửi riêng thư mục output nêu trên cùng bộ notebook bàn giao.
